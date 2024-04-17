import umdalib
from pathlib import Path as pt

hiddenimports = ["umdalib.server", "umdalib.getVersion"]
icons_dir = pt(umdalib.__file__).parent / "../icons"
icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]

datas = icons_files
