from umdalib.logger import logger
import json
import pathlib
import datetime
import decimal
import numpy as np
import types
from pathlib import Path as pt


def convert_to_json_compatible(obj):
    if isinstance(obj, dict):
        return {key: convert_to_json_compatible(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_compatible(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_compatible(item) for item in obj)
    elif isinstance(obj, set):
        return list(convert_to_json_compatible(item) for item in obj)
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    elif isinstance(obj, types.FunctionType):
        return f"<function {obj.__name__}>"
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        if hasattr(obj, "__dict__"):
            return {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "attributes": convert_to_json_compatible(obj.__dict__),
            }
        else:
            return str(obj)


def safe_json_dump(
    obj: dict, filename: str | pt, overwrite=True, create_dir: bool = True
):
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dictionary, got {type(obj)}")

    if not isinstance(filename, (str, pt)):
        raise ValueError(f"Expected a string or Path, got {type(filename)}")

    if isinstance(filename, str):
        filename = pt(filename)

    if filename.suffix != ".json":
        filename = filename.with_suffix(".json")

    if filename.exists():
        if overwrite:
            filename.unlink()
            logger.warning(f"Deleted existing file: {filename} to overwrite")
        else:
            logger.error(f"File already exists: {filename}")
            raise FileExistsError(f"File already exists: {filename}")

    if not filename.parent.exists() and create_dir:
        logger.warning(f"Creating directory: {filename.parent}")
        filename.parent.mkdir(parents=True)

    try:
        logger.info(f"Saving to {filename}")
        with open(filename, "w") as f:
            json.dump(convert_to_json_compatible(obj), f, indent=4)
            logger.success(f"{filename.name} saved successfully to {filename.parent}")
    except Exception as e:
        logger.error(f"Error saving to {filename}: {e}")
        raise e
