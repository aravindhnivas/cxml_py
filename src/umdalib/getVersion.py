import sys
from multiprocessing import cpu_count

from umdalib import __version__ as umdalib_version
from umdalib.utils import NPARTITIONS, RAM_IN_GB

# from umdalib.utils import logger


def main(args=None):
    version_info = sys.version_info
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    return {
        "python": version,
        "umdapy": umdalib_version,
        "cpu_count": cpu_count(),
        "ram": RAM_IN_GB,
        "npartitions": NPARTITIONS,
    }
