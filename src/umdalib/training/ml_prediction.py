from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path as pt
from typing import TypedDict

import numpy as np
import pandas as pd
from joblib import load

from umdalib.training.embedd_data import get_smi_to_vec
from umdalib.utils import logger


@dataclass
class Args:
    smiles: str
    pretrained_model_file: str
    test_file: str


@lru_cache()
def load_model():
    if not pretrained_model_file:
        raise ValueError("Pretrained model file not found")
    return load(pretrained_model_file)


def predict_from_file(test_file: pt, smi_to_vector, model, estimator, scaler):
    logger.info(f"Reading test file: {test_file}")
    data = pd.read_csv(test_file)
    logger.info(f"Data shape: {data.shape}")

    columns = data.columns.tolist()
    if len(columns) == 0:
        raise ValueError(
            "Test file should have at least one column with header name SMILES"
        )

    if columns[0] != "SMILES":
        raise ValueError("Test file should have a column header named 'SMILES'")

    smiles = data["SMILES"].tolist()
    X = [smi_to_vector(smi, model) for smi in smiles]
    logger.info(f"X shape: {len(X)}")
    if len(X) == 0:
        raise ValueError("No valid SMILES found in test file")

    predicted_value: np.ndarray = estimator.predict(X)

    if scaler:
        predicted_value = scaler.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()

    predicted_value = predicted_value.tolist()
    data["predicted_value"] = predicted_value
    savefile = (
        test_file.parent
        / f"{test_file.stem}_predicted_values_{pretrained_model_file.stem}.csv"
    )
    data.to_csv(savefile, index=False)

    logger.info(f"Predicted values saved to {savefile}")
    return {"savedfile": str(savefile)}


pretrained_model_file = None


def main(args: Args):
    global pretrained_model_file

    pretrained_model_file = pt(args.pretrained_model_file)
    pretrained_model_loc = pretrained_model_file.parent

    arguments_file = (
        pretrained_model_loc / f"{pretrained_model_file.stem}.arguments.json"
    )
    if not arguments_file.exists():
        raise ValueError(f"Arguments file not found: {arguments_file}")

    with open(arguments_file, "r") as f:
        arguments = json.load(f)
        logger.info(
            f"Arguments: {arguments} from {arguments_file} for {pretrained_model_file} loaded"
        )

    vectors_file = pt(arguments["vectors_file"])
    vectors_metadata_file = vectors_file.parent / f"{vectors_file.stem}.metadata.json"
    if not vectors_metadata_file.exists():
        raise ValueError(f"Vectors metadata file not found: {vectors_metadata_file}")

    vectors_metadata = None
    with open(vectors_metadata_file, "r") as f:
        vectors_metadata = json.load(f)
        logger.info(
            f"Vectors metadata: {vectors_metadata} loaded from {vectors_metadata_file}"
        )

    if not vectors_metadata:
        raise ValueError(f"Vectors metadata not found in {vectors_metadata_file}")

    # logger.info(f"Parsing SMILES: {args.smiles}")

    predicted_value = None
    estimator = None

    smi_to_vector, model = get_smi_to_vec(
        vectors_metadata["embedder"],
        vectors_metadata["pre_trained_embedder_location"],
        vectors_metadata["PCA_location"],
    )

    logger.info(f"Loading estimator from {pretrained_model_file}")
    estimator, scaler = load_model()

    if not estimator:
        logger.error("Failed to load estimator")
        raise ValueError("Failed to load estimator")

    logger.info(f"Loaded estimator: {estimator}")
    logger.info(f"Loaded scaler: {scaler}")

    if args.test_file:
        return predict_from_file(
            pt(args.test_file), smi_to_vector, model, estimator, scaler
        )

    X = smi_to_vector(args.smiles, model)
    logger.info(f"X: {X}")
    predicted_value: np.ndarray = estimator.predict([X])

    if scaler:
        predicted_value = scaler.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()

    predicted_value = float(predicted_value[0])
    logger.info(f"Predicted value: {predicted_value}")

    return {"predicted_value": f"{predicted_value:.2f}"}
