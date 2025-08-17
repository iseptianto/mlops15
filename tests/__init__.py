"""
fastapibaru: FastAPI service untuk inference model tourism.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fastapibaru")  # kalau dipasang pakai `pip -e .`
except PackageNotFoundError:
    __version__ = "0.0.0"

# Optional: re-export objek app jika file utama `app.py`/`main.py`
try:
    # pilih salah satu sesuai file kamu
    from .app import app  # noqa: F401
except Exception:
    # fallback ke main.py kalau app.py tidak ada
    try:
        from .main import app  # noqa: F401
    except Exception:
        # biarkan diam—nanti diimport oleh uvicorn langsung via modul
        pass
