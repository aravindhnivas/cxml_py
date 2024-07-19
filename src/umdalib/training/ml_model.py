from dataclasses import dataclass
from typing import Dict, Union
from umdalib.utils import logger


@dataclass
class Args:
    model: str
    vectors_file: str
    labels_file: str
    bootstrap: str
    bootstrap_nsamples: int
    parameters: Dict[str, Union[str, int]]


def main(args: Args):
    logger.info(f"Training {args.model} model")
