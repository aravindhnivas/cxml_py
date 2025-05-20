from umdalib import molecule_analysis
from pathlib import Path as pt

loc = pt(molecule_analysis.__file__).parent
hiddenimports = [
    f"umdalib.molecule_analysis.{file.stem}"
    for file in loc.glob("*.py")
    if file.stem != "__init__"
]
print(f"{hiddenimports=}\ndynamically generated...")
