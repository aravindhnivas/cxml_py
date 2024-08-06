from dataclasses import dataclass

import numpy as np
from umdalib.utils import logger
from typing import TypedDict

from umdalib.training.embedd_data import get_smi_to_vec
from joblib import load
from functools import lru_cache


class Embedder(TypedDict):
    name: str
    file: str
    pipeline_file: str


@dataclass
class Args:
    smiles: str
    molecular_embedder: Embedder
    pretrained_model_file: str


@lru_cache()
def load_model(model_location: str):
    return load(model_location)


def main(args: Args):

    logger.info(f"Parsing SMILES: {args.smiles}")

    predicted_value = None
    estimator = None

    smi_to_vector, model = get_smi_to_vec(
        args.molecular_embedder["name"],
        args.molecular_embedder["file"],
        args.molecular_embedder["pipeline_file"],
    )

    X = smi_to_vector(args.smiles, model)
    logger.info(f"X: {X}")
    return {"predicted_value": 1}

    logger.info(f"Loading estimator from {args.pretrained_model_file}")
    estimator, scaler = load_model(args.pretrained_model_file)
    if not estimator:
        logger.error("Failed to load estimator")
        raise ValueError("Failed to load estimator")

    logger.info(f"Loaded estimator: {estimator}")
    logger.info(f"Loaded scaler: {scaler}")

    predicted_value: np.ndarray = estimator.predict([X])

    if scaler:
        predicted_value = scaler.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()

    predicted_value = float(predicted_value[0])
    logger.info(f"Predicted value: {predicted_value}")

    return {"predicted_value": f"{predicted_value:.2f}"}
