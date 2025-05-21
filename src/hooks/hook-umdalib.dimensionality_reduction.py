from umdalib import dimensionality_reduction
from pathlib import Path as pt

loc = pt(dimensionality_reduction.__file__).parent
hiddenimports = [
    f"umdalib.dimensionality_reduction.{file.stem}"
    for file in loc.glob("*.py")
    if file.stem != "__init__"
]
print(f"{hiddenimports=}\ndynamically generated...")
