from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
# from astrochem_embedding import VICGAE
from pathlib import Path as pt
import numpy as np
from rdkit import Chem
from mol2vec import features
from umdalib.utils import load_model
from umdalib.utils import logger

# from loguru import logger

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
    pretrained_model_location: str


def VICGAE2vec(smi: str):
    global invalid_smiles
    smi = str(smi).replace("\xa0", "")
    if smi == "nan":
        return None
    # model = VICGAE.from_pretrained()
    try:
        return VICGAE_model.embed_smiles(smi).numpy()
    except:
        invalid_smiles.append(smi)
        return np.zeros((1, 32))


mol2vec_model = None
VICGAE_model = None
invalid_smiles = []

def mol2vec(smi: str) -> list[np.ndarray]:
    """
    Given a model, convert a SMILES string into the corresponding
    NumPy vector.
    """

    global invalid_smiles
    smi = str(smi).replace("\xa0", "")

    if smi == "nan":
        return None

    # Molecule from SMILES will break on "bad" SMILES; this tries
    # to get around sanitization (which takes a while) if it can
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if not mol:
            if not isinstance(smi, str):
                return None
            invalid_smiles.append(str(smi))

        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        # generate a sentence from rdkit molecule
        sentence = features.mol2alt_sentence(mol, radius=1)
        # generate vector embedding from sentence and model
        vector = features.sentences2vec([sentence], mol2vec_model)
        
        return vector

    except:
        if smi not in invalid_smiles and isinstance(smi, str):
            invalid_smiles.append(smi)
            
        return np.zeros((1, mol2vec_model.vector_size))


embedding_model: dict[str, Callable] = {
    "VICGAE": VICGAE2vec,
    "mol2vec": mol2vec,
}


def main(args: Args):

    logger.info(f"{args=}")

    global invalid_smiles, mol2vec_model, VICGAE_model

    fullfile = pt(args.filename)
    location = fullfile.parent
    logger.info(f"Reading {fullfile} as {args.filetype}")

    if args.embedding == "mol2vec":
        mol2vec_model = load_model(args.pretrained_model_location)
        logger.info(f"Loaded mol2vec model with {mol2vec_model.vector_size} dimensions")
    elif args.embedding == "VICGAE":
        VICGAE_model = load_model(args.pretrained_model_location, use_joblib=True)
        logger.info(f"Loaded VICGAE model")
        
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

    vectors: dd = df[args.df_column].apply(apply_model, meta=(None, np.float32))
    # vectors: dd = df[args.df_column].apply(apply_model, meta=pd.DataFrame({0: pd.Series(dtype='float64')}))

    if vectors is None:
        raise ValueError(f"Unknown embedding model: {args.embedding}")

    embedd_savefile = f"{fullfile.stem}_{args.df_column}_{args.embedding}.npy"
    logger.info(f"Begin computing embeddings for {fullfile.stem}...")
    time = perf_counter()

    start_time = perf_counter()
    computed_time = None
    with ProgressBar():
        vec_computed = vectors.compute()
        computed_time = f"{(perf_counter() - start_time):.2f} s"
        np.save(location / embedd_savefile, vec_computed)

    logger.info(f"{len(vec_computed[0])=}, {vec_computed[0]=}")

    logger.info(
        f"Embeddings computed in {(perf_counter() - time):.2f} s and saved to {embedd_savefile}"
    )

    # \xa0 is a non-breaking space in Latin1 (ISO 8859-1), also known as NBSP in Unicode. It's a character that prevents an automatic line break at its position. In HTML, it's often used to create multiple spaces that are visible.
    invalid_smiles = [
        smiles.replace("\xa0", "").strip()
        for smiles in invalid_smiles
        # if isinstance(smiles, str)
    ]

    return {
        "name": embedd_savefile,
        "shape": vec_computed.shape[0],
        "invalid_smiles": invalid_smiles,
        "saved_file": f"{location / embedd_savefile}",
        "computed_time": computed_time,
    }
