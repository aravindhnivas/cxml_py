from umdalib import load_file
from pathlib import Path as pt

loc = pt(load_file.__file__).parent
hiddenimports = [
    f"umdalib.load_file.{file.stem}"
    for file in loc.glob("*.py")
    if file.stem != "__init__"
]
print(f"{hiddenimports=}\ndynamically generated...")
