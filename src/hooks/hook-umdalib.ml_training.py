from umdalib import ml_training
from pathlib import Path as pt

loc = pt(ml_training.__file__).parent
hiddenimports = [
    f"umdalib.ml_training.{file.stem}"
    for file in loc.glob("*.py")
    if file.stem != "__init__"
]
print(f"{hiddenimports=}\ndynamically generated...")
