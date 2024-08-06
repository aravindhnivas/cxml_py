from dataclasses import dataclass
from umdalib.utils import load_model, logger
from typing import TypedDict
from umdalib.training.embedd_data import get_smi_to_vec


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

    smi_to_vector, model = get_smi_to_vec(
        args.molecular_embedder["name"],
        args.molecular_embedder["file"],
        args.molecular_embedder["pipeline_file"],
    )

    X = smi_to_vector(args.smiles, model)
    logger.info(f"X: {X}")

    estimator = load_model(args.pretrained_model_file, use_joblib=True)
    logger.info(f"Loaded estimator: {estimator}")

    predicted_value = estimator.predict(X)
    logger.info(f"Predicted value: {predicted_value}")

    return {"predicted_value": predicted_value}
