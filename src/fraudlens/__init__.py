"""FraudLens: explainable real-time credit card fraud detection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fraudlens")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
