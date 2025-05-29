from dataclasses import dataclass
import json
from typing import Literal
import joblib
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.pipeline import Pipeline
import umap
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
import phate
import trimap
from cxml_lib.logger import logger
from pathlib import Path as pt
from cxml_lib.utils import parse_args
from cxml_lib.utils.json import safe_json_dump
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Args:
    params: dict
    vector_file: str
    dr_savefile: str
    embedder_loc: str
    method: Literal[
        "PCA",
        "UMAP",
        "t-SNE",
        "KernelPCA",
        "PHATE",
        "ISOMAP",
        "LaplacianEigenmaps",
        "TriMap",
        "FactorAnalysis",
    ] = "PCA"
    embedder_name: str = "mol2vec"
    scaling: bool = True
    save_diagnostics: bool = True
    diagnostics_file: str = "dr_diagnostics.json"


# Making the PHATE transformer compatible with sklearn
class PHATETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.phate = None

    def fit(self, X, y=None):
        self.phate = phate.PHATE(**self.kwargs)
        self.phate.fit(X)
        return self

    def transform(self, X):
        return self.phate.transform(X)


def save_diagnostics_to_json(
    method: str, reducer, reduced: np.ndarray, diagnostics_file: str
):
    diagnostics = {
        "method": method,
        "reduced_shape": list(reduced.shape),
    }

    if method in ["PCA", "FactorAnalysis"]:
        if hasattr(reducer, "explained_variance_ratio_"):
            diagnostics["explained_variance_ratio"] = (
                reducer.explained_variance_ratio_.tolist()
            )
            diagnostics["cumulative_variance"] = np.cumsum(
                reducer.explained_variance_ratio_
            ).tolist()
    elif method in ["KernelPCA"]:
        if hasattr(reducer, "lambdas_"):
            diagnostics["kernel_eigenvalues"] = reducer.lambdas_.tolist()
    elif method in ["t-SNE", "UMAP", "PHATE", "ISOMAP", "LaplacianEigenmaps", "TriMap"]:
        # Generic notes
        diagnostics["info"] = (
            f"{method} does not provide variance info. Saved 2D shape only."
        )

    with open(diagnostics_file, "w") as f:
        json.dump(diagnostics, f, indent=4)
    logger.info(f"Saved diagnostics to {diagnostics_file}")


def main(args: Args):
    logger.info("Starting dimensionality reduction pipeline")
    args = parse_args(args.__dict__, Args)
    # logger.info(json.dumps(args.__dict__, indent=4))

    logger.info(f"Loading embeddings from {args.vector_file}")

    # return

    # Load data
    X: np.ndarray = np.load(args.vector_file, allow_pickle=True)

    logger.info(f"{X.shape=}")

    logger.info(f"Applying {args.method} with parameters: {args.params}")

    # Apply dimensionality reduction
    if args.method == "PCA":
        reducer = PCA(**args.params)
    elif args.method == "UMAP":
        reducer = umap.UMAP(**args.params)
    elif args.method == "t-SNE":
        reducer = TSNE(**args.params)
    elif args.method == "KernelPCA":
        reducer = KernelPCA(**args.params)
    elif args.method == "PHATE":
        # reducer = phate.PHATE(**args.params)
        reducer = PHATETransformer(**args.params)
    elif args.method == "ISOMAP":
        reducer = Isomap(**args.params)
    elif args.method == "LaplacianEigenmaps":
        reducer = SpectralEmbedding(**args.params)
    elif args.method == "TriMap":
        reducer = trimap.TRIMAP(**args.params)
    elif args.method == "FactorAnalysis":
        reducer = FactorAnalysis(**args.params)
    else:
        logger.error(f"Unsupported method: {args.method}")
        raise ValueError(f"Unsupported method: {args.method}")

    steps = []
    if args.scaling:
        steps.append(("scaler", StandardScaler()))

    # Add the DR method
    steps.append(("reducer", reducer))
    pipeline = Pipeline(steps)
    reduced = pipeline.fit_transform(X)

    # zeroing out the reduced vector if the original vector is zero i.e., invalid embedding
    zero_vec_ind: np.ndarray[bool] = np.all(X == 0, axis=1)
    reduced[zero_vec_ind] = np.zeros(
        reduced.shape[1]
    )  # Set the reduced vector to zero if the original vector is zero
    logger.info(f"Reduced data shape: {reduced.shape}")
    if zero_vec_ind.any():
        logger.info(
            f"First reduced vector for all-zero input: {reduced[zero_vec_ind][0]}"
        )
    else:
        logger.info("No all-zero vectors found in the original embedding.")

    np.save(args.dr_savefile, reduced)
    logger.info(f"Saved reduced data to {args.dr_savefile}")

    dr_savefile = pt(args.dr_savefile)
    # invalid_smiles = []
    invalid_vec_ind: np.ndarray[bool] = np.all(reduced == 0, axis=1)
    invalid_smiles = []

    if invalid_vec_ind.sum() > 0:
        invalid_smiles = reduced[invalid_vec_ind]

    save_obj = {
        "data_shape": reduced.shape,
        "invalid_smiles": len(invalid_smiles),
    }
    safe_json_dump(save_obj, dr_savefile.with_suffix(".metadata.json"))

    # save_loc = dr_savefile.parent / args.method.lower()

    pipeline_loc = dr_savefile.parent / "dr_pipelines"
    pipeline_loc.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, pipeline_loc / f"{dr_savefile.stem}.joblib")
    logger.info("Saved DR pipeline to dr_pipeline.joblib")

    save_diagnostics_file = (
        dr_savefile.parent / "dr_diagnostics" / f"{dr_savefile.stem}.diagnostics.json"
    )

    save_diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving diagnostics to {save_diagnostics_file}")
    if args.save_diagnostics:
        save_diagnostics_to_json(args.method, reducer, reduced, save_diagnostics_file)
