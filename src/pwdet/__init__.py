"""pwdet: poisoned-weights detector for LLM checkpoints and endpoints."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("pwdet")
except _metadata.PackageNotFoundError:  # running from a tree that was never installed
    __version__ = "0.2.0"
