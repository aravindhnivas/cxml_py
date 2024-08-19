from dataclasses import dataclass
import json
import pandas as pd
from umdalib.training.read_data import read_as_ddf
from umdalib.utils import logger
from collections import Counter
import multiprocessing
from pathlib import Path as pt


@dataclass
class Args:
    analysis_file: str
    min_atomic_number: int
    max_atomic_number: int
    size_count_threshold: int
    elemental_count_threshold: int
    filter_elements: list[str]
    filter_structures: list[str]


def main(args: Args):

    analysis_file = pt(args.analysis_file)
    df = pd.read_csv(analysis_file)
    df["ElementCategories"] = df["ElementCategories"].apply(
        lambda x: Counter(json.loads(x))
    )
    df["Elements"] = df["Elements"].apply(lambda x: Counter(json.loads(x)))
