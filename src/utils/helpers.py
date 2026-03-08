"""General utility helpers shared across modules."""
from __future__ import annotations

import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timer(fn: Callable) -> Callable:
    """Decorator that logs execution time in milliseconds."""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        from config.logging_config import logger
        logger.debug("%s completed in %.1f ms", fn.__qualname__, elapsed_ms)
        return result
    return wrapper


def extract_ab_choice(text: str) -> str | None:
    """Extract the first 'a' or 'b' from LLM response text (case-insensitive)."""
    match = re.search(r"\b([ab])\b", text.strip().lower())
    return match.group(1) if match else None


def chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into sublists of at most `size` elements."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]
