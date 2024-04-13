from pathlib import Path as pt
import tempfile
from os import environ
from platform import system
from multiprocessing import cpu_count
from loguru import logger
from psutil import virtual_memory
from gensim.models import word2vec
import umdapy

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


logger.add(
    Paths().app_log_dir / "umdapy_server.log",
    rotation="10 MB",
    compression="zip",
)


umdapy_dir = pt(umdapy.__file__).parent
model_dir = umdapy_dir / "models"
logger.info(f"umdapy_dir: {umdapy_dir}")
logger.info(f"model_dir: {model_dir}")


def load_model(filepath: str):
    filepath = model_dir / filepath

    logger.info(f"Loading model from {filepath}")
    if not filepath.exists():
        logger.error(f"Model file not found: {filepath}")
        raise FileNotFoundError(f"Model file not found: {filepath}")
    logger.info(f"Model loaded from {filepath}")
    return word2vec.Word2Vec.load(str(filepath))
