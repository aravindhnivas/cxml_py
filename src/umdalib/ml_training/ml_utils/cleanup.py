import shutil
from tempfile import gettempdir
import os


def cleanup_temp_files():
    temp_dir = gettempdir()
    for item in os.listdir(temp_dir):
        if item.startswith("joblib_memmapping_folder_"):
            try:
                shutil.rmtree(os.path.join(temp_dir, item))
            except (OSError, IOError):
                pass
    print("Temp files cleaned up")


# Call this before starting new operations
# cleanup_temp_files()
