from pathlib import Path as pt
import tempfile
from os import environ
from platform import system
from multiprocessing import cpu_count
from psutil import virtual_memory

RAM_IN_GB = virtual_memory().total / 1024**3
NPARTITIONS = cpu_count() * 5
BUNDLE_IDENTIFIER = "com.umdaui.dev"


class Paths:
    def get_app_log_dir(self):
        # Linux: Resolves to ${configDir}/${bundleIdentifier}/logs.
        # macOS: Resolves to ${homeDir}/Library/Logs/{bundleIdentifier}
        # Windows: Resolves to ${configDir}/${bundleIdentifier}/logs.

        if system() == "Linux":
            return pt(environ["HOME"]) / ".config" / BUNDLE_IDENTIFIER / "logs"
        elif system() == "Darwin":
            return pt(environ["HOME"]) / "Library" / "Logs" / BUNDLE_IDENTIFIER
        elif system() == "Windows":
            return pt(environ["APPDATA"]) / BUNDLE_IDENTIFIER / "logs"
        else:
            raise NotImplementedError(f"Unknown system: {system()}")

    def get_temp_dir(self):
        return pt(tempfile.gettempdir()) / BUNDLE_IDENTIFIER

    @property
    def app_log_dir(self):
        return self.get_app_log_dir()

    @property
    def temp_dir(self):
        return self.get_temp_dir()
