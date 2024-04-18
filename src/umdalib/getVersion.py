import sys
import umdalib
from multiprocessing import cpu_count
from umdalib.utils import NPARTITIONS, RAM_IN_GB

# from umdalib.utils import logger


def main(args=""):
    version_info = sys.version_info
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    return {
        "python": version,
        "umdapy": umdalib.__version__,
        "cpu_count": cpu_count(),
        "ram": RAM_IN_GB,
        "npartitions": NPARTITIONS,
    }
