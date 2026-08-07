import importlib.metadata

from semeio.semeio import setup_logging, valid_file

try:  # ruff: ignore[suppressible-exception, non-empty-init-module]
    __version__ = importlib.metadata.distribution("semeio").version
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    pass

setup_logging()  # ruff: ignore[non-empty-init-module]

__all__ = ["valid_file"]
