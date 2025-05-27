from typing import List
import numpy as np
from rdkit import Chem
from mol2vec import features
from cxml_lib.logger import logger
from pathlib import Path as pt
import joblib
from gensim.models import word2vec
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
import torch
import pandas as pd
import mapply
from tqdm import tqdm

mapply.init(n_workers=-1, chunk_size=100, max_chunks_per_worker=10, progressbar=True)


def VICGAE2vec(df: pd.Series | str, model):
    def func(smi: str):
        smi = str(smi).replace("\xa0", "")
        if smi == "nan":
            return np.zeros(32)
        try:
            return model.embed_smiles(smi).numpy().reshape(-1)
        except Exception as _:
            return np.zeros(32)

    if isinstance(df, str):
        return func(df)

    return df.mapply(func).to_numpy()


"""
mol2vec is inspired and derived from Kelvin Lee's UMDA repository:
https://github.com/laserkelvin/umda/blob/a95cc1c1eb98a1ff64b37a4c6ed92f9546d0215b/umda/smi_vec.py
"""


def mol2vec(df: pd.Series | str, model, radius=1) -> List[np.ndarray]:
    def func(smi: str):
        smi = str(smi).replace("\xa0", "")
        if smi == "nan":
            return np.zeros(model.vector_size)

        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            sentence = features.mol2alt_sentence(mol, radius)
            vector = features.sentences2vec([sentence], model)
            vector = vector.reshape(-1)
            if len(vector) == 1:
                logger.error(f"Vector length is {len(vector)} for {smi}")
                raise ValueError(f"Invalid embedding: {smi}")
            return vector

        except Exception as _:
            return np.zeros(model.vector_size)

    if isinstance(df, str):
        return func(df)

    # for some reason, mapply is not faster with mol2vec embeddings
    return df.apply(func).to_numpy()


def molecules_to_vectors_using_huggingface(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    smiles: list[str],
    batch_size: int = 64,
    device: str = "cpu",  # or "cuda" if GPU is available
):
    model = model.to(device)
    all_embeddings = []
    progress = 0
    for i in tqdm(range(0, len(smiles), batch_size), desc="Embedding SMILES"):
        batch = smiles[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # you can experiment with mean, max, or CLS:
            # embeds = out.last_hidden_state.mean(dim=1).cpu().detach().numpy()
            embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeds)

        # current progress for batch
        progress += len(batch) / len(smiles) * 100
        logger.info(f"Progress: {progress:.2f}%")

    all_embeddings = np.vstack(all_embeddings)
    all_embeddings = all_embeddings.squeeze()
    logger.info(f"{all_embeddings.shape=}")
    return all_embeddings


hf_model: PreTrainedModel | None = None
hf_tokenizer: PreTrainedTokenizer | None = None


def hf_func(df: pd.Series | str, model=None):
    global hf_model, hf_tokenizer
    if isinstance(df, str):
        smiles = [df]
    else:
        smiles = df.tolist()
    return molecules_to_vectors_using_huggingface(hf_model, hf_tokenizer, smiles)


def get_smi_to_vec(embedding, pretrained_file):
    global hf_model, hf_tokenizer

    logger.info(f"Loading model from {pretrained_file}")
    if not pt(pretrained_file).exists():
        logger.error(f"Model file not found: {pretrained_file}")
        raise FileNotFoundError(f"Model file not found: {pretrained_file}")
    logger.info(f"Model loaded from {pretrained_file}")

    model = None
    hf_model = None
    hf_tokenizer = None
    smi_to_vector = None

    if embedding == "mol2vec":
        model = word2vec.Word2Vec.load(str(pretrained_file))
        logger.info(f"Loaded mol2vec model with {model.vector_size} dimensions")
        smi_to_vector = mol2vec
    elif embedding == "VICGAE":
        model = joblib.load(pretrained_file)
        logger.info("Loaded VICGAE model")
        smi_to_vector = VICGAE2vec
    elif embedding == "ChemBERTa-zinc-base-v1":
        # smi_to_vector = ChemBERTa_to_vec
        hf_model = AutoModel.from_pretrained(
            pretrained_file,
            trust_remote_code=True,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_file,
            trust_remote_code=True,
        )
        # smi_to_vector = hf_func
    elif embedding == "MoLFormer-XL-both-10pct":
        hf_model = AutoModel.from_pretrained(
            pretrained_file,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_file,
            trust_remote_code=True,
        )
        # smi_to_vector = hf_func
    else:
        raise ValueError(f"Unknown embedding model: {embedding}")

    if smi_to_vector is None and embedding in [
        "MoLFormer-XL-both-10pct",
        "ChemBERTa-zinc-base-v1",
    ]:
        logger.info(f"Using HuggingFace model for {embedding}")
        if hf_model is None or hf_tokenizer is None:
            raise ValueError(
                f"HuggingFace model or tokenizer not found for {embedding}"
            )
        smi_to_vector = hf_func

    if smi_to_vector is None:
        raise ValueError(f"Unknown embedding model: {embedding}")

    logger.warning(f"{smi_to_vector=}")
    if not callable(smi_to_vector):
        raise ValueError(f"Unknown embedding model: {embedding}")

    return smi_to_vector, model
