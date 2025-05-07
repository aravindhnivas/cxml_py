from umdalib import embedder
from pathlib import Path as pt

loc = pt(embedder.__file__).parent
hiddenimports = [
    f"umdalib.embedder.{file.stem}"
    for file in loc.glob("*.py")
    if file.stem != "__init__"
]
print(f"{hiddenimports=}\ndynamically generated...")
