import json
from dataclasses import dataclass
from pathlib import Path as pt

import numpy as np
import pandas as pd
from joblib import load

from umdalib.training.embedd_data import smi_to_vec_dict
from umdalib.logger import logger
import joblib
from gensim.models import word2vec


@dataclass
class Args:
    smiles: str
    pretrained_model_file: str
    prediction_file: str
    embedder_name: str
    embedder_loc: str


def load_embedder(embedder_name: str, embedder_loc: str, use_joblib: bool = False):
    if not embedder_loc:
        raise Exception("Embedder location not provided")

    logger.info(f"Loading model from {embedder_loc}")
    if not pt(embedder_loc).exists():
        logger.error(f"Model file not found: {embedder_loc}")
        raise FileNotFoundError(f"Model file not found: {embedder_loc}")
    logger.info(f"Model loaded from {embedder_loc}")

    if use_joblib:
        return joblib.load(embedder_loc)

    if embedder_name == "mol2vec":
        return word2vec.Word2Vec.load(str(embedder_loc))
    elif embedder_name == "VICGAE":
        return joblib.load(embedder_loc)
    else:
        raise Exception(f"Unsupported embedder: {embedder_name}")


def load_model():
    if not pretrained_model_file:
        raise ValueError("Pretrained model file not found")
    return load(pretrained_model_file)


def predict_from_file(
    prediction_file: pt,
    vectors_file: pt,
    smi_to_vector,
    embedder_model,
    estimator,
    scaler,
):
    if not prediction_file.exists():
        raise ValueError(f"Prediction file not found: {prediction_file}")

    logger.info(f"Reading test file: {prediction_file}")

    # smiles = np.loadtxt(prediction_file, dtype=str, delimiter="\n")
    with open(prediction_file, "r") as f:
        smiles = f.read().splitlines()

    logger.info(f"Read {len(smiles)} SMILES from {prediction_file}")

    if len(smiles) == 0:
        raise ValueError("No valid SMILES found in test file")

    X = np.array([smi_to_vector(smi, embedder_model) for smi in smiles])
    logger.info(f"{X.shape=}")

    if "_with" in vectors_file.stem:
        dr_pipeline = joblib.load(
            vectors_file.parent / "dr_pipelines" / f"{vectors_file.stem}.joblib"
        )
        X: np.ndarray = dr_pipeline.transform([X])
        X = np.squeeze(X)
        logger.info(f"Transformed X shape: {X.shape=}")
        logger.info(f"{X.shape=}")

    if len(X) == 0:
        raise ValueError("No valid SMILES found in test file")

    predicted_value: np.ndarray = estimator.predict(X)

    if scaler:
        predicted_value = scaler.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()

    predicted_value = predicted_value.tolist()

    data = pd.DataFrame({"SMILES": smiles})
    data["predicted_value"] = predicted_value
    savefile = (
        prediction_file.parent
        / f"{prediction_file.stem}_predicted_values_{pretrained_model_file.stem}.csv"
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

    if not arguments:
        raise ValueError(f"{arguments_file=} is invalid or empty")

    vectors_file = arguments.get("vectors_file")
    if not vectors_file:
        raise ValueError(f"{vectors_file=} is invalid or empty")

    vectors_file = pt(vectors_file)

    predicted_value = None
    estimator = None

    embedder_model = load_embedder(
        embedder_name=args.embedder_name, embedder_loc=args.embedder_loc
    )
    smi_to_vector = smi_to_vec_dict[args.embedder_name]

    logger.info(f"Loading estimator from {pretrained_model_file}")
    estimator, scaler = load_model()

    if not estimator:
        logger.error("Failed to load estimator")
        raise ValueError("Failed to load estimator")

    logger.info(f"Loaded estimator: {estimator}")
    logger.info(f"Loaded scaler: {scaler}")

    if args.prediction_file:
        prediction_file = pt(args.prediction_file)
        return predict_from_file(
            prediction_file,
            vectors_file,
            smi_to_vector,
            embedder_model,
            estimator,
            scaler,
        )

    logger.info(f"Loading smi: {args.smiles}")
    X = smi_to_vector(args.smiles, embedder_model)
    logger.info(f"{X.shape=}")

    if "_with" in vectors_file.stem:
        dr_pipeline = joblib.load(
            vectors_file.parent / "dr_pipelines" / f"{vectors_file.stem}.joblib"
        )
        X: np.ndarray = dr_pipeline.transform([X])
        X = np.squeeze(X)
        logger.info(f"Transformed X shape: {X.shape=}")
        logger.info(f"{X.shape=}")

    predicted_value: np.ndarray = estimator.predict([X])

    if scaler:
        predicted_value = scaler.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()

    predicted_value = float(predicted_value[0])
    logger.info(f"Predicted value: {predicted_value}")
    return {"predicted_value": predicted_value}
