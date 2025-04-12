from dataclasses import dataclass, fields
import json
from pathlib import Path as pt
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umdalib.logger import logger


@dataclass
class Args:
    params: dict
    processed_df_file: str
    dr_file: str
    embedder_loc: str
    method: Literal["PCA", "UMAP", "t-SNE"] = "PCA"
    embedder_name: str = "mol2vec"
    scaling: bool = True


def parge_args(args_dict: dict) -> Args:
    # Build kwargs using defaults where necessary
    kwargs = {}
    for field in fields(Args):
        if field.name in args_dict:
            kwargs[field.name] = args_dict[field.name]
        elif field.default is not field.default_factory:  # default value exists
            kwargs[field.name] = field.default
        elif field.default_factory is not None:  # default factory (rare case)
            kwargs[field.name] = field.default_factory()
        else:
            raise ValueError(f"Missing required field: {field.name}")
    return Args(**kwargs)


def main(args: Args):
    logger.info("Starting dimensionality reduction pipeline")
    args = parge_args(args.__dict__)
    logger.info(json.dumps(args.__dict__, indent=4))

    logger.info(f"Loading embeddings from {args.processed_df_file}")

    # return

    # Load data
    processed_df = pd.read_parquet(args.processed_df_file)
    X = processed_df.iloc[:, 2:].to_numpy()
    # y = processed_df["y"].to_numpy()
    logger.info(f"{X.shape=}")
    # Load embedder (if needed for future extensions)

    logger.info(f"Applying {args.method} with parameters: {args.params}")

    # Apply dimensionality reduction
    if args.method == "PCA":
        reducer = PCA(**args.params)
    elif args.method == "UMAP":
        reducer = umap.UMAP(**args.params)
    elif args.method == "t-SNE":
        reducer = TSNE(**args.params)
    else:
        logger.error(f"Unsupported method: {args.method}")
        raise ValueError(f"Unsupported method: {args.method}")

    if args.scaling:
        scalar = StandardScaler()
        X = scalar.fit_transform(X)

    reduced = reducer.fit_transform(X)
    logger.info(f"{reduced.shape=}")
    # reduced = reducer.fit_transform(X)

    # Save output
    save_path = pt(args.dr_file)
    np.save(save_path, reduced)

    logger.info(f"Saved reduced data to {args.dr_file}")
