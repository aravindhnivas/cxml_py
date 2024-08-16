from dataclasses import dataclass
import numpy as np
import pandas as pd
from umdalib.utils import logger
from typing import TypedDict
from umdalib.training.embedd_data import get_smi_to_vec
from joblib import load
from functools import lru_cache
from pathlib import Path as pt


class Embedder(TypedDict):
    name: str
    file: str
    pipeline_file: str


@dataclass
class Args:
    smiles: str
    molecular_embedder: Embedder
    pretrained_model_file: str
    test_file: str
    use_dask: bool


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

    if not args.pretrained_model_file:
        raise ValueError("Pretrained model file not found")

    pretrained_model_file = pt(args.pretrained_model_file)

    logger.info(f"Parsing SMILES: {args.smiles}")

    predicted_value = None
    estimator = None

    smi_to_vector, model = get_smi_to_vec(
        args.molecular_embedder["name"],
        args.molecular_embedder["file"],
        args.molecular_embedder["pipeline_file"],
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
