from PyInstaller.utils.hooks import collect_submodules

# hiddenimports = [
#     "pygsp.features",
#     "pygsp.optimization",
#     "pygsp.filters",
# ]
hiddenimports = collect_submodules("pygsp")
