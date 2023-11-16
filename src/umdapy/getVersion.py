import sys
import umdapy


def main(args=""):
    version_info = sys.version_info
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    print(f"Python {version} (umdapy {umdapy.__version__})")
    return {"python": version, "umdapy": umdapy.__version__}