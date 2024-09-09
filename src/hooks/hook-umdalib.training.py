from umdalib import training
from pathlib import Path as pt

loc = pt(training.__file__).parent
hiddenimports = [
    f"umdalib.training.{file.stem}"
    for file in loc.glob("*.py")
    if file.stem != "__init__"
]
print(f"{hiddenimports=}\ndynamically generated...")
