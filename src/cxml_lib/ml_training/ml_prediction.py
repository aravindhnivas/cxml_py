import json
from dataclasses import dataclass
from pathlib import Path as pt
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from cxml_lib.vectorize_molecules.embedder import get_smi_to_vec
from cxml_lib.logger import logger
import joblib
from gensim.models import word2vec
from scipy.special import inv_boxcox
from cxml_lib.utils.json import safe_json_load


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


pretrained_model_file = None


def inverse_transform_data(data: np.ndarray, method: str):
    if method == "log1p":
        return np.expm1(data)
    elif method == "sqrt":
        return np.power(data, 2)
    elif method == "reciprocal":
        return (1 / data) - 1
    elif method == "square":
        return np.sqrt(data)
    elif method == "exp":
        return np.log(data)
    else:
        raise ValueError(f"Unsupported transformation method: {method}")


def predict_value(
    X: np.ndarray,
    arguments: dict[str, Any],
):
    logger.info(f"Loading estimator from {pretrained_model_file}")
    estimator, yscaler_old = load_model()
    logger.info(f"Loaded estimator: {estimator}")

    if not estimator:
        logger.error("Failed to load estimator")
        raise ValueError("Failed to load estimator")

    yscaling = arguments.get("yscaling")
    yscaler = None
    yscaler_file = pretrained_model_file.with_suffix(".yscaler.pkl")

    if yscaling:
        if yscaler_file.exists():
            yscaler = load(yscaler_file)
        else:
            if yscaler_old:
                yscaler = yscaler_old
                logger.info(f"Using old yscaler: {yscaler}")
            else:
                raise ValueError("Yscaler file not found and no old yscaler found")

        logger.info(f"Loaded yscaler: {yscaler}")

    ytransformation = arguments.get("ytransformation")
    y_transformer = None
    y_transformer_file = pretrained_model_file.with_suffix(".y_transformer.pkl")

    if ytransformation:
        if not y_transformer_file.exists():
            raise ValueError("Y_transformer file not found")
        y_transformer = load(y_transformer_file)
        logger.info(f"Loaded y_transformer: {y_transformer}")

    if yscaling and yscaler is None:
        raise ValueError("Yscaler not found")

    if ytransformation and y_transformer is None:
        raise ValueError("Y_transformer not found")

    predicted_value: np.ndarray = estimator.predict(X)
    logger.info(f"Predicted value: {predicted_value}")
    logger.info(f"yscaler: {yscaler}")

    if yscaler:
        predicted_value = yscaler.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()
        logger.info(f"Scaled predicted value: {predicted_value}")

    if ytransformation == "boxcox":
        boxcox_lambda_param_file = pretrained_model_file.with_suffix(
            ".boxcox_lambda_param.json"
        )
        boxcox_lambda_param = safe_json_load(boxcox_lambda_param_file)
        if boxcox_lambda_param is None:
            raise ValueError("Boxcox lambda parameter not found")
        predicted_value = inv_boxcox(predicted_value, boxcox_lambda_param)
        logger.info(f"Boxcox inverse transformed predicted value: {predicted_value}")
    elif ytransformation == "yeo_johnson":
        if y_transformer is None:
            raise ValueError("Y_transformer not found")
        predicted_value = y_transformer.inverse_transform(
            predicted_value.reshape(-1, 1)
        ).flatten()
        logger.info(
            f"Yeo-Johnson inverse transformed predicted value: {predicted_value}"
        )
    elif ytransformation:
        predicted_value = inverse_transform_data(predicted_value, ytransformation)
        logger.info(f"Inverse transformed predicted value: {predicted_value}")

    return predicted_value


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
        arguments: dict[str, Any] = json.load(f)
        logger.info(
            f"Arguments: {arguments} from {arguments_file} for {pretrained_model_file} loaded"
        )

    if not arguments:
        raise ValueError(f"{arguments_file=} is invalid or empty")

    vectors_file = arguments.get("vectors_file")
    if not vectors_file:
        raise ValueError(f"{vectors_file=} is invalid or empty")

    vectors_file = pt(vectors_file)

    embedder_model = load_embedder(
        embedder_name=args.embedder_name, embedder_loc=args.embedder_loc
    )
    smi_to_vector, embedder_model = get_smi_to_vec(
        args.embedder_name, args.embedder_loc
    )

    prediction_file = None
    if args.prediction_file:
        prediction_file = pt(args.prediction_file)
        with open(args.prediction_file, "r") as f:
            smiles = f.read().splitlines()

        logger.info(f"Read {len(smiles)} SMILES from {prediction_file}")

        if len(smiles) == 0:
            raise ValueError("No valid SMILES found in test file")

        X = np.array([smi_to_vector(smi, embedder_model) for smi in smiles])
    else:
        X = smi_to_vector(args.smiles, embedder_model)
        X = np.array([X])
        logger.info(f"Loaded X: {X}")

    if "_with" in vectors_file.stem:
        dr_pipeline = joblib.load(
            vectors_file.parent / "dr_pipelines" / f"{vectors_file.stem}.joblib"
        )
        X: np.ndarray = dr_pipeline.transform(X)
        X = np.squeeze(X)
        logger.info(f"Transformed X shape: {X.shape=}")
        logger.info(f"{X.shape=}")

    predicted_value = predict_value(
        X,
        arguments,
    )

    if prediction_file is not None:
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
    else:
        return {"predicted_value": float(predicted_value[0])}
