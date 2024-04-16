import umdapy
from pathlib import Path as pt

hiddenimports = ["umdapy.server", "umdapy.getVersion"]
icons_dir = pt(umdapy.__file__).parent / "../icons"
models_dir = pt(umdapy.__file__).parent / "models"

icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]
models_files = [(str(file.resolve()), "models") for file in models_dir.glob("*")]

datas = icons_files + models_files
