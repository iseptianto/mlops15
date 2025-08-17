"""
modelbaru: modul training & utilities (preprocess, train, eval, log ke MLflow).
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("modelbaru")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Pamerkan API yang sering dipakai
try:
    from .train import train_pipeline, build_model, preprocess  # noqa: F401
except Exception:
    pass
