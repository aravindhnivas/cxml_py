from dataclasses import dataclass
from pathlib import Path as pt
from time import perf_counter
from typing import Literal, Union
import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar
from umdalib.load_file.read_data import read_as_ddf
from umdalib.utils.json import safe_json_dump
from umdalib.logger import logger
import pandas as pd

from umdalib.vectorize_molecules.vectorizer import get_smi_to_vec
import mapply

mapply.init(n_workers=-1, chunk_size=100, max_chunks_per_worker=10, progressbar=True)


def convert_to_float(value: Union[str, float]) -> float:
    try:
        return float(value)
    except ValueError:
        if isinstance(value, str) and "-" in value:
            parts = value.split("-")
            if len(parts) == 2 and parts[0] and parts[1]:
                try:
                    return (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    pass
        return np.nan


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    npartitions: int
    columnX: str
    columnY: str
    embedding: Literal[
        "VICGAE", "mol2vec", "ChemBERTa-zinc-base-v1", "MoLFormer-XL-both-10pct"
    ]
    pretrained_model_location: str
    test_mode: bool
    test_smiles: str
    embedd_savefile: str
    vectors_file: str
    use_dask: bool
    index_col: str


test_mode = False


def main(args: Args):
    logger.info(f"{args=}")

    global test_mode

    test_mode = args.test_mode

    smi_to_vector, model = get_smi_to_vec(
        args.embedding, args.pretrained_model_location
    )

    if test_mode:
        logger.info(f"Testing with {args.test_smiles}")

        vec: np.ndarray = smi_to_vector(args.test_smiles, model)
        vec = vec.squeeze()
        logger.info(f"{vec.shape=}\n")

        return {
            "test_mode": {
                "embedded_vector": vec.tolist() if vec is not None else None,
            }
        }

    fullfile = pt(args.filename)
    logger.info(f"Reading {fullfile} as {args.filetype}")

    ddf = read_as_ddf(args.filetype, args.filename, args.key, use_dask=args.use_dask)

    if args.index_col:
        ddf = ddf.set_index(args.index_col)

    if args.use_dask:
        logger.info(f"{args.npartitions=}")
        ddf = ddf.repartition(npartitions=args.npartitions)

    vectors = None
    logger.info(f"Using {args.embedding} for embedding")

    logger.info(f"Using {smi_to_vector} for embedding")
    if not callable(smi_to_vector):
        raise ValueError(f"Unknown embedding model: {args.embedding}")

    if args.use_dask:
        vectors = ddf[args.columnX].apply(
            smi_to_vector, args=(model,), meta=(None, np.float32)
        )
    else:
        # for some reason, mapply is not faster with mol2vec embeddings
        # if args.embedding == "mol2vec":
        #     vectors = ddf[args.columnX].apply(smi_to_vector, args=(model,))
        # else:
        #     vectors = ddf[args.columnX].mapply(smi_to_vector, args=(model,))
        vectors = smi_to_vector(ddf[args.columnX], model)

    if vectors is None:
        raise ValueError(f"Unknown embedding model: {args.embedding}")

    vectors_file = pt(args.vectors_file)
    embedding_loc = vectors_file.parent
    if not embedding_loc.exists():
        embedding_loc.mkdir(parents=True)

    logger.info(f"{vectors_file=}")
    logger.info(f"Begin computing embeddings for {fullfile.stem}...")
    time = perf_counter()
    y = ddf[args.columnY]
    y = y.apply(convert_to_float)
    vec_computed: np.ndarray = None

    with ProgressBar():
        if args.use_dask:
            if isinstance(vectors, da.Array):
                vec_computed = vectors.compute()
            y = y.compute()
        else:
            vec_computed = vectors

        # logger.info(f"{vec_computed.shape=}")
        vec_computed = np.vstack(vec_computed)

        np.save(vectors_file, vec_computed)
        logger.success(f"Embedded numpy array saved to {vectors_file}")

    logger.info(f"{vec_computed.shape=}")

    invalid_indices_full = []
    invalid_vec_ind: np.ndarray[bool] = np.all(vec_computed == 0, axis=1)
    invalid_smiles_filename = vectors_file.with_suffix(".invalid_embeddings.csv")
    invalid_smiles = []

    if invalid_vec_ind.sum() > 0:
        invalid_indices_full.extend(["# Invalid embeddings"])
        invalid_smiles_df: pd.Series = ddf[args.columnX].loc[invalid_vec_ind]
        invalid_smiles = invalid_smiles_df.values
        invalid_smiles_df.to_csv(invalid_smiles_filename)
        invalid_indices_full.extend(invalid_smiles_df.index.values)

    invalid_y_ind = y.isna()
    if invalid_y_ind.sum() > 0:
        invalid_indices_full.extend(["# Invalid y values"])
        invalid_y = y[invalid_y_ind]
        invalid_y_filename = vectors_file.with_suffix(".invalid_y.csv")
        invalid_y.to_csv(invalid_y_filename)
        logger.info(f"Invalid Y values saved to {invalid_y_filename}")
        invalid_indices_full.extend(invalid_y.index.values)

    if len(invalid_indices_full) > 0:
        # invalid_indices = np.append(invalid_smiles.index.values, invalid_y.index.values)
        invalid_indices_full_filename = vectors_file.with_suffix(".invalid_indices.txt")
        np.savetxt(invalid_indices_full_filename, invalid_indices_full, fmt="%s")
        logger.info(f"Invalid indices saved to {invalid_indices_full_filename}")
    else:
        # save an empty file called all valid
        np.savetxt(vectors_file.with_suffix(".all_valid"), [], fmt="%s")
        logger.info("No invalid indices found")
    logger.info(
        f"Embeddings computed in {(perf_counter() - time):.2f} s and saved to {vectors_file.name}"
    )

    save_obj = {
        "embedder": args.embedding,
        "pre_trained_embedder_location": args.pretrained_model_location,
        "filename": args.filename,
        "filetype": args.filetype,
        "key": args.key,
        "npartitions": args.npartitions,
        "columnX": args.columnX,
        "data_shape": vec_computed.shape,
        "invalid_smiles": len(invalid_smiles),
        "invalid_smiles_file": str(invalid_smiles_filename),
    }
    safe_json_dump(save_obj, vectors_file.with_suffix(".metadata.json"))

    return {
        "file_mode": {
            "name": vectors_file.name,
            "shape": vec_computed.shape[0],
            "invalid_smiles": len(invalid_smiles),
            "invalid_smiles_file": str(invalid_smiles_filename),
            "saved_file": str(vectors_file),
        }
    }
