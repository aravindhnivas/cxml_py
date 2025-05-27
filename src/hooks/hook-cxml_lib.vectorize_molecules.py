from cxml_lib import vectorize_molecules
from pathlib import Path as pt

loc = pt(vectorize_molecules.__file__).parent
hiddenimports = [
    f"cxml_lib.vectorize_molecules.{file.stem}"
    for file in loc.glob("*.py")
    if file.stem != "__init__"
]
print(f"{hiddenimports=}\ndynamically generated...")
