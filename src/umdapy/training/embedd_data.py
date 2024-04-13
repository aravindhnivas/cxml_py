from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from astrochem_embedding import VICGAE
from pathlib import Path as pt
import numpy as np
from rdkit import Chem
from mol2vec import features
from umdapy.utils import load_model, logger


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    npartitions: int
    mol2vec_dim: int
    PCA_dim: int
    df_column: str
    embedding: Literal["VICGAE", "mol2vec"]


def VICGAE2vec(smi: str):
    global invalid_smiles
    model = VICGAE.from_pretrained()
    try:
        return model.embed_smiles(smi).numpy()
    except:
        invalid_smiles.append(smi)
        return None


mol2vec_model = load_model("mol2vec/mol2vec_model.pkl")
logger.info(f"Loaded mol2vec model with {mol2vec_model.vector_size} dimensions")
invalid_smiles = []


def mol2vec(smi: str) -> list[np.ndarray]:
    """
    Given a model, convert a SMILES string into the corresponding
    NumPy vector.
    """

    global invalid_smiles

    # Molecule from SMILES will break on "bad" SMILES; this tries
    # to get around sanitization (which takes a while) if it can
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if not mol:
        invalid_smiles.append(smi)
        return None

    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    # generate a sentence from rdkit molecule
    sentence = features.mol2alt_sentence(mol, radius=1)
    # generate vector embedding from sentence and model
    vector = features.sentences2vec([sentence], mol2vec_model)
    return vector


embedding_model: dict[str, Callable] = {
    "VICGAE": VICGAE2vec,
    "mol2vec": mol2vec,
}


def main(args: Args):

    fullfile = pt(args.filename)
    location = fullfile.parent

    logger.info(f"Reading {fullfile} as {args.filetype}")
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

    logger.info(f"{args.npartitions=}")
    df = df.repartition(npartitions=args.npartitions)

    vectors = None
    logger.info(f"Using {args.embedding} for embedding")
    apply_model = embedding_model[args.embedding]
    logger.info(f"Using {apply_model} for embedding")
    if not callable(apply_model):
        raise ValueError(f"Unknown embedding model: {args.embedding}")

    vectors = df[args.df_column].apply(apply_model, meta=(None, "object"))

    if vectors is None:
        raise ValueError(f"Unknown embedding model: {args.embedding}")

    embedd_savefile = f"{fullfile.stem}_{args.df_column}_{args.embedding}.npy"
    logger.info(f"Begin computing embeddings for {fullfile.stem}...")
    time = perf_counter()

    with ProgressBar():
        vec_computed = vectors.compute()
        np.save(location / embedd_savefile, vec_computed)

    logger.info(f"{vec_computed[0]=}, {vec_computed[0].shape=}")

    logger.info(
        f"Embeddings computed in {(perf_counter() - time):.2f} s and saved to {embedd_savefile}"
    )

    return {
        "name": embedd_savefile,
        "shape": vec_computed.shape[0],
        "invalid_smiles": invalid_smiles,
        "saved_file": f"{location / embedd_savefile}",
    }
