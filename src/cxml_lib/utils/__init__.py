from multiprocessing import cpu_count
from dataclasses import fields
from typing import Any
from psutil import virtual_memory

RAM_IN_GB = virtual_memory().total / 1024**3
NPARTITIONS = cpu_count() * 5


def parse_args(args_dict: dict, args_class: Any) -> Any:
    kwargs = {}
    for field in fields(args_class):
        if field.name in args_dict:
            kwargs[field.name] = args_dict[field.name]
        elif field.default is not field.default_factory:
            kwargs[field.name] = field.default
        elif field.default_factory is not None:
            kwargs[field.name] = field.default_factory()
        else:
            raise ValueError(f"Missing required field: {field.name}")
    return args_class(**kwargs)
