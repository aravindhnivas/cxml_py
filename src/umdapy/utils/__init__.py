from pathlib import Path as pt
import tempfile


def logger(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def get_temp_dir():
    return pt(tempfile.gettempdir()) / "com.umdaui.dev"
