# extra-hooks/hook-wandb.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
datas = collect_data_files(
    "wandb", include_py_files=True, includes=["**/vendor/**/*.py"]
)

# These are imports from vendored stuff in wandb
hiddenimports += collect_submodules("pathtools")
