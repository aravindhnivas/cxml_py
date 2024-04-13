from dataclasses import dataclass
from time import perf_counter
from typing import Literal
import dask.dataframe as dd
from loguru import logger
from dask.diagnostics import ProgressBar
from astrochem_embedding import VICGAE
from multiprocessing import cpu_count
from pathlib import Path as pt
import numpy as np

NPARTITIONS = cpu_count() * 5

# pbar = ProgressBar()
# pbar.register()

embedding_model = None


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    df_column: str
    embedding: Literal["VICGAE", "mol2vec"]


def embedding_to_vec(smi):
    try:
        return embedding_model.embed_smiles(smi).numpy()
    except:
        return None


def main(args: Args):
    global embedding_model

    fullfile = pt(args.filename)
    location = fullfile.parent

    print(f"Reading {fullfile} as {args.filetype}")
    df = None
    if args.filetype == "csv":
        df = dd.read_csv(fullfile)
    elif args.filetype == "parquet":
        df = dd.read_parquet(fullfile)
    elif args.filetype == "hdf":
        df = dd.read_hdf(fullfile, args.key)
    elif args.filetype == "json":
        df = dd.read_json(fullfile)
    else:
        raise ValueError(f"Unknown filetype: {args.filetype}")

    logger.info(f"{NPARTITIONS=}")
    df = df.repartition(npartitions=NPARTITIONS)

    if args.embedding == "VICGAE":
        embedding_model = VICGAE.from_pretrained()

        vectors = df[args.df_column].apply(embedding_to_vec, meta=(None, "object"))

        embedd_savefile = f"{fullfile.stem}_{args.df_column}_embedded.npy"
        logger.info(f"Begin computing embeddings for {fullfile.stem}...")
        time = perf_counter()
        with ProgressBar():
            vec_computed = vectors.compute()
            np.save(location / embedd_savefile, vec_computed)

        logger.info(
            f"Embeddings computed in {(perf_counter() - time):.2f} s and saved to {embedd_savefile}"
        )
    return {"name": embedd_savefile, "shape": vec_computed.shape[0]}
