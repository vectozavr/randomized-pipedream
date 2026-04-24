from __future__ import annotations

import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.objectives.base import Objective
from src.utils.batching import Batch


# =========================================================================
# HELPER FUNCTIONS: NumPy <-> PyTorch Bridge
# =========================================================================
def set_module_weights(module: nn.Module, flat_weights: np.ndarray) -> None:
    """Loads a flat numpy array into a PyTorch module's parameters on the correct device."""
    offset = 0
    # Auto-detect which device this module is on
    device = next(module.parameters()).device
    with torch.no_grad():
        for param in module.parameters():
            numel = param.numel()
            # Push the numpy data to the same device as the module
            param_data = torch.from_numpy(flat_weights[offset: offset + numel]).to(device).view_as(param)
            param.copy_(param_data)
            offset += numel


def get_module_weights(module: nn.Module) -> np.ndarray:
    """Extracts a PyTorch module's parameters into a flat numpy array on CPU."""
    with torch.no_grad():
        return np.concatenate([p.cpu().numpy().flatten() for p in module.parameters()])


def get_module_grads(module: nn.Module) -> np.ndarray:
    """Extracts a PyTorch module's gradients into a flat numpy array on CPU."""
    grads = []
    for p in module.parameters():
        if p.grad is not None:
            grads.append(p.grad.cpu().numpy().flatten())
        else:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(grads)


# =========================================================================
# PYTORCH LLM MODULES
# =========================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Causal mask is handled natively by PyTorch's scaled_dot_product_attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =========================================================================
# THE PIPELINE OBJECTIVE
# =========================================================================
class SimpleLLMObjective(Objective):
    def __init__(
            self,
            text_data: str,
            num_stages: int,
            batch_size: int,
            seq_len: int = 8,
            embed_dim: int = 32,
            num_heads: int = 2,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_pipeline_stages = num_stages

        if num_stages < 3:
            raise ValueError("SimpleLLM requires at least 3 stages: Embedding, Block(s), Output.")

        # -------------------------------------------------------------
        # CHECK FOR GPU (A100 will be detected here)
        # -------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SimpleLLMObjective initialized on device: {self.device}")

        # 1. Prepare Character Dataset
        chars = sorted(list(set(text_data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.data = np.array([self.stoi[ch] for ch in text_data], dtype=np.int64)

        # Build batches (X, y)
        self._batches = []
        num_batches = (len(self.data) - 1) // (batch_size * seq_len)
        for i in range(num_batches):
            idx = i * batch_size * seq_len
            Xb = self.data[idx: idx + batch_size * seq_len].reshape(batch_size, seq_len)
            yb = self.data[idx + 1: idx + 1 + batch_size * seq_len].reshape(batch_size, seq_len)
            self._batches.append((Xb, yb))

        # 2. Build Pipeline Stages as isolated PyTorch Modules
        self.stages_modules = []

        # Stage 0: Token + Positional Embedding
        class EmbeddingStage(nn.Module):
            def __init__(self, vocab_size, embed_dim, seq_len):
                super().__init__()
                self.tok_emb = nn.Embedding(vocab_size, embed_dim)
                self.pos_emb = nn.Embedding(seq_len, embed_dim)

            def forward(self, x):
                B, T = x.size()
                pos = torch.arange(0, T, dtype=torch.long, device=x.device)
                return self.tok_emb(x) + self.pos_emb(pos)

        self.stages_modules.append(EmbeddingStage(self.vocab_size, embed_dim, seq_len))

        # Stage 1 to N-2: Transformer Blocks
        for _ in range(num_stages - 2):
            self.stages_modules.append(TransformerBlock(embed_dim, num_heads))

        # Stage N-1: Output Head
        class OutputStage(nn.Module):
            def __init__(self, embed_dim, vocab_size):
                super().__init__()
                self.ln_f = nn.LayerNorm(embed_dim)
                self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

            def forward(self, x):
                return self.lm_head(self.ln_f(x))

        self.stages_modules.append(OutputStage(embed_dim, self.vocab_size))

        # Move all modules to GPU and set to train mode
        for m in self.stages_modules:
            m.to(self.device)
            m.train()

    @property
    def num_stages(self) -> int:
        return self.num_pipeline_stages

    def initial_activation(self, batch: Batch) -> np.ndarray:
        return np.zeros(1)

    def initial_stage_weights(self, mode: str = "random", seed: int = 0, scale=None) -> list[np.ndarray]:
        torch.manual_seed(seed)
        return [get_module_weights(m) for m in self.stages_modules]

    def get_batches(self) -> list[Batch]:
        return self._batches

    # ---------------------------------------------------------------------
    # PIPELINE FORWARD PASS
    # ---------------------------------------------------------------------
    def forward_stage(self, batch: Batch, stage: int, w_stage: np.ndarray, activation_in: np.ndarray) -> tuple[
        np.ndarray, dict]:
        module = self.stages_modules[stage]
        set_module_weights(module, w_stage)

        with torch.no_grad():
            if stage == 0:
                Xb, _ = batch
                # Move input batch to GPU
                x_tensor = torch.tensor(Xb, dtype=torch.long, device=self.device)
            else:
                # Move incoming activations to GPU
                x_tensor = torch.tensor(activation_in, dtype=torch.float32, device=self.device)

            out_tensor = module(x_tensor)

        cache = {"activation_in": activation_in.copy()}

        # Pull output activations back to CPU as NumPy array
        return out_tensor.cpu().numpy(), cache

    # ---------------------------------------------------------------------
    # LOSS & GRADIENT
    # ---------------------------------------------------------------------
    def loss_and_output_grad(self, batch: Batch, final_activation: np.ndarray) -> tuple[float, np.ndarray]:
        _, yb = batch
        # Push activations and targets to GPU
        logits = torch.tensor(final_activation, dtype=torch.float32, requires_grad=True, device=self.device)
        targets = torch.tensor(yb, dtype=torch.long, device=self.device)

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        loss.backward()

        # Pull gradients back to CPU
        return loss.item(), logits.grad.cpu().numpy()

    # ---------------------------------------------------------------------
    # PIPELINE BACKWARD PASS (Uses Activation Recomputation)
    # ---------------------------------------------------------------------
    def backward_stage(self, batch: Batch, stage: int, w_stage: np.ndarray, cache: dict, grad_out: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        module = self.stages_modules[stage]

        set_module_weights(module, w_stage)
        module.zero_grad()

        if stage == 0:
            Xb, _ = batch
            x_tensor = torch.tensor(Xb, dtype=torch.long, device=self.device)
        else:
            x_tensor = torch.tensor(cache["activation_in"], dtype=torch.float32, requires_grad=True, device=self.device)

        out_tensor = module(x_tensor)

        # Move incoming backward gradients to GPU
        grad_out_tensor = torch.tensor(grad_out, dtype=torch.float32, device=self.device)
        out_tensor.backward(grad_out_tensor)

        grad_w = get_module_grads(module)

        if stage == 0:
            grad_in = np.zeros_like(grad_out)
        else:
            # Pull backwards activations to CPU
            grad_in = x_tensor.grad.cpu().numpy()

        return grad_w, grad_in

    # =====================================================================
    # FULL EVALUATION HELPERS
    # =====================================================================
    def full_objective(self, stage_weights: list[np.ndarray]) -> float:
        total_loss = 0.0
        for batch in self._batches:
            act = None
            for stage in range(self.num_stages):
                module = self.stages_modules[stage]
                set_module_weights(module, stage_weights[stage])
                with torch.no_grad():
                    if stage == 0:
                        act = module(torch.tensor(batch[0], dtype=torch.long, device=self.device))
                    else:
                        act = module(act)
            loss, _ = self.loss_and_output_grad(batch, act.cpu().numpy())
            total_loss += loss
        return total_loss / len(self._batches)

    def full_gradient(self, stage_weights: list[np.ndarray]) -> list[np.ndarray]:
        dummy_grads = []
        for module in self.stages_modules:
            num_params = sum(p.numel() for p in module.parameters())
            dummy_grads.append(np.zeros(num_params, dtype=np.float32))
        return dummy_grads

    @property
    def optimal_objective_value(self) -> float:
        return 0.0

    @property
    def smoothness_constant(self) -> float:
        return 1.0

    @property
    def stage_slices(self) -> list[slice]:
        slices = []
        offset = 0
        for module in self.stages_modules:
            length = sum(p.numel() for p in module.parameters())
            slices.append(slice(offset, offset + length))
            offset += length
        return slices