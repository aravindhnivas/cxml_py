from dataclasses import dataclass
from umdalib.utils import load_model, logger
from typing import TypedDict


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
    molecular_embedder = load_model(args.molecular_embedder_file)
    pretrained_model = load_model(args.pretrained_model_file, use_joblib=True)

    predicted_value = None
    return {"predicted_value": predicted_value}
