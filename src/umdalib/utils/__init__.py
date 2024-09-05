import tempfile
from multiprocessing import cpu_count
from os import environ
from pathlib import Path as pt
from platform import system

import joblib
from gensim.models import word2vec
from loguru import logger
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


logfile = Paths().app_log_dir / "umdapy_server.log"
logger.info(f"Logging to {logfile}")

logger.add(
    logfile,
    rotation="10 MB",
    compression="zip",
)

logger.info(f"loaded joblib: {joblib.__version__}")


def load_model(filepath: str, use_joblib: bool = False):
    logger.info(f"Loading model from {filepath}")
    if not pt(filepath).exists():
        logger.error(f"Model file not found: {filepath}")
        raise FileNotFoundError(f"Model file not found: {filepath}")
    logger.info(f"Model loaded from {filepath}")

    if use_joblib:
        return joblib.load(filepath)
    return word2vec.Word2Vec.load(str(filepath))
