from dataclasses import dataclass

import numpy as np
from umdalib.utils import logger
from typing import TypedDict

from umdalib.training.embedd_data import get_smi_to_vec
from joblib import load


class Embedder(TypedDict):
    name: str
    file: str
    pipeline_file: str


@dataclass
class Args:
    smiles: str
    molecular_embedder: Embedder
    pretrained_model_file: str


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

    logger.info(f"Loading estimator from {args.pretrained_model_file}")
    estimator, scaler = load(args.pretrained_model_file)
    if not estimator:
        logger.error("Failed to load estimator")
        raise ValueError("Failed to load estimator")

    logger.info(f"Loaded estimator: {estimator}")
    logger.info(f"Loaded scaler: {scaler}")
    # return {"predicted_value": 1}

    predicted_value: np.ndarray = estimator.predict([X])

    if scaler:
        predicted_value = scaler.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()

    logger.info(f"Predicted value: {predicted_value}")

    return {"predicted_value": predicted_value}
