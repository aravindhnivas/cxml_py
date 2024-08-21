from dataclasses import dataclass
from mol2vec import features
from pathlib import Path as pt
from loguru import logger

from umdalib.utils import Paths

logger.add(
    Paths().app_log_dir / "mol2vec.log",
    rotation="10 MB",
    compression="zip",
)


def gen_corpus(
    smi_file: str,
    radius: int = 1,
    sentence_type: str = "alt",
):
    smi_in_file: pt = pt(smi_file)

    corpus_out_file = (
        smi_in_file.parent
        / f"m2v_corpus_radius_{radius}_sentence_type_{sentence_type}_smi_{smi_in_file.stem}.dat"
    )

    features.generate_corpus(
        in_file=str(smi_in_file.resolve()),
        out_file=str(corpus_out_file.resolve()),
        r=radius,
        sentence_type=sentence_type,
        n_jobs=n_jobs,
    )
    return corpus_out_file


def gen_model(
    corpus_file_in: pt,
    vector_size: int = 300,
    min_count: int = 1,
):
    if isinstance(corpus_file_in, str):
        corpus_file_in = pt(corpus_file_in)

    pkl_out = (
        corpus_file_in.parent / f"m2v_model_{vector_size}-dim_{min_count}-min-count.pkl"
    )

    model = features.train_word2vec_model(
        infile_name=str(corpus_file_in.resolve()),
        outfile_name=str(pkl_out.resolve()),
        vector_size=vector_size,
        min_count=min_count,
        n_jobs=n_jobs,
    )

    return model, pkl_out


n_jobs = 1  # Number of cpu cores used for calculation
# warning >1 CPU core may cause memory errors
# Need to fix this in mol2vec package


@dataclass
class Args:
    smi_file: str
    corpus_file: str
    radius: int
    vector_size: int
    sentence_type: str
    n_jobs: int
    min_count: int


def main(args: Args):
    global n_jobs
    logger.info("#" * 80 + "\n\n")
    logger.info(f"\n\nStarting Mol2Vec model generation with {args=}\n\n")

    # if args.n_jobs:
    #     n_jobs = args.n_jobs

    logger.info(f"Generating Mol2Vec model from SMILES file. Using {n_jobs} CPU cores.")
    logger.info(f"SMILES file: {args.smi_file}")

    if not pt(args.smi_file).exists():
        logger.error(f"SMILES file {args.smi_file} does not exist.")
        raise FileNotFoundError(f"SMILES file {args.smi_file} does not exist.")

    # return

    m2v_model = None
    pkl_file: str = None

    if args.corpus_file:
        logger.info(f"Using existing corpus file: {args.corpus_file}")
        m2v_model, pkl_file = gen_model(
            args.corpus_file, args.vector_size, args.min_count
        )
    else:
        smi_file = pt(args.smi_file)

        logger.info(f"Generating corpus file from {smi_file}")
        corpus_file = gen_corpus(smi_file, args.radius, args.sentence_type)
        logger.success(f"Corpus file saved to {corpus_file}")

        logger.info(f"Generating Mol2Vec model from corpus file {corpus_file}")
        m2v_model, pkl_file = gen_model(corpus_file, args.vector_size, args.min_count)

    if not pt(pkl_file).exists():
        logger.error(f"Model file {pkl_file} does not exist.")
        raise FileNotFoundError(f"Model file {pkl_file} does not exist.")

    if not m2v_model:
        logger.error("Model not generated.")
        raise ValueError("Model not generated.")

    logger.success(
        f"Model save to {pkl_file} with vector size {m2v_model.vector_size} dimensions."
    )
    logger.info("#" * 80 + "\n\n")
