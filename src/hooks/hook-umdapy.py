import umdapy
from pathlib import Path as pt

hiddenimports = ["umdapy.server", "umdapy.getVersion"]
icons_dir = pt(umdapy.__file__).parent / "../icons"
icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]

datas = icons_files
