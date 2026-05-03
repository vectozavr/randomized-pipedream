from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


TOY_CORPUS = """
PipeDream moves microbatches through a row of stages.
GPD reads a deliberately stale mixture of stage weights.
This tiny language-model corpus repeats enough structure for a smoke experiment:
stage zero embeds tokens, middle stages transform them, and the final stage predicts the next character.
The comparison is not about benchmark accuracy; it is about making the same optimizer abstractions touch a richer objective.
"""


@dataclass(frozen=True)
class TextDatasetInfo:
    source: str
    path: Path | None
    num_chars: int
    vocab_size: int


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _repeat_to_length(text: str, min_chars: int) -> str:
    if len(text) >= min_chars:
        return text
    repeats = (min_chars // max(1, len(text))) + 1
    return (text + "\n") * repeats


def _download_text(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=30) as response:
            raw = response.read()
    except URLError as exc:
        raise RuntimeError(
            f"Could not download dataset from {url}. "
            f"Use --dataset toy or provide --dataset-file instead. Original error: {exc}"
        ) from exc

    text = raw.decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return text


def load_text_dataset(
    *,
    dataset: str,
    data_dir: Path,
    dataset_file: Path | None = None,
    min_chars: int = 1,
    max_chars: int | None = None,
    force_download: bool = False,
) -> tuple[str, TextDatasetInfo]:
    """Load a char-level text dataset for SimpleLLMObjective.

    Supported sources:
    - "toy": a deterministic handcrafted string, repeated as needed.
    - "tiny_shakespeare": downloaded once into data_dir.
    - "file": a user-provided local text file.
    """
    if min_chars <= 1:
        min_chars = 2

    path: Path | None = None
    if dataset == "toy":
        text = TOY_CORPUS
    elif dataset == "tiny_shakespeare":
        path = data_dir / "tiny_shakespeare.txt"
        if force_download or not path.exists():
            text = _download_text(TINY_SHAKESPEARE_URL, path)
        else:
            text = path.read_text(encoding="utf-8")
    elif dataset == "file":
        if dataset_file is None:
            raise ValueError("--dataset-file is required when --dataset file is used.")
        path = dataset_file
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    text = _normalize_text(text)
    text = _repeat_to_length(text, min_chars)
    if max_chars is not None:
        if max_chars < min_chars:
            raise ValueError("max_chars must be >= the minimum characters required by the experiment.")
        text = text[:max_chars]

    if len(set(text)) < 2:
        raise ValueError("The text dataset must contain at least two distinct characters.")

    return text, TextDatasetInfo(
        source=dataset,
        path=path,
        num_chars=len(text),
        vocab_size=len(set(text)),
    )
