from PyInstaller.utils.hooks import collect_all

# Add mpl-data directory (required for fonts and style definitions)
import matplotlib
import os

# Only include the backends you actually use
hiddenimports = [
    "matplotlib.backends.backend_pdf",  # For PDF saving
    "matplotlib.backends.backend_agg",  # For PNG saving (uses agg backend)
    "matplotlib.pyplot",
    "numpy",  # Required dependency
]

# Collect matplotlib data files (needed for basic functionality)
datas, binaries, _ = collect_all("matplotlib")


mpl_data_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data")
datas.extend(
    [
        (mpl_data_dir, "matplotlib/mpl-data"),
    ]
)
