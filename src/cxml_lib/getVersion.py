import sys
from multiprocessing import cpu_count
from cxml_lib import __version__ as pyPackageVersion
from cxml_lib.utils import NPARTITIONS, RAM_IN_GB


def main(args=None):
    version_info = sys.version_info
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    return {
        "python": version,
        "pyPackageVersion": pyPackageVersion,
        "cpu_count": cpu_count(),
        "ram": RAM_IN_GB,
        "npartitions": NPARTITIONS,
    }
