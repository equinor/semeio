import importlib.metadata

from semeio.semeio import setup_logging, valid_file

try:  # noqa: SIM105, RUF067
    __version__ = importlib.metadata.distribution("semeio").version
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    pass

setup_logging()  # noqa: RUF067

__all__ = ["valid_file"]
