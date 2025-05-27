from multiprocessing import cpu_count
from psutil import virtual_memory

RAM_IN_GB = virtual_memory().total / 1024**3
NPARTITIONS = cpu_count() * 5
