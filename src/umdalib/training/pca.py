from time import sleep
import numpy as np
import h5py
from pathlib import Path as pt
from umdalib.utils import load_model
import numpy as np

USE_DASK = True
from dask import array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client

if USE_DASK:
    from dask_ml.decomposition import IncrementalPCA
    from dask_ml.cluster import KMeans
    from dask_ml.preprocessing import StandardScaler
else:
    from sklearn.decomposition import IncrementalPCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

from umdalib.utils import logger
from joblib import load, dump, parallel_backend
from sklearn.pipeline import make_pipeline
from .embedd_data import smi_to_vec_dict


def train_fit_model(data: np.ndarray, model: IncrementalPCA):
    """
    This function just helps simplify the main code by handling various contexts.
    If `dask` is being used, we use the dask backend for computation as well
    as making sure that the result is actually computed.
    """
    if USE_DASK:
        backend = "dask"
    else:
        backend = "threading"
    with parallel_backend(backend, n_jobs=n_workers):
        model.fit(data)
        transform = model.transform(data)
        if USE_DASK:
            transform = transform.compute()
        # if we are fitting a clustering model we grab the labels
        labels = getattr(model, "labels_", None)
        if USE_DASK and labels is not None:
            labels = labels.compute()

    return (model, transform, labels)


def generate_embeddings():

    with h5py.File(h5_file, "w") as embeddings_file:

        np_vec = np.load(npy_file, allow_pickle=True)
        vectors = np.vstack(np_vec)
        embeddings_file["vectors"] = vectors

        if USE_DASK:
            client = Client(threads_per_worker=threads_per_worker, n_workers=n_workers)
            vectors = da.from_array(np.vstack(np_vec))

        try:

            scaler = StandardScaler()
            pca_model = IncrementalPCA(n_components=pca_dim)
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

            # preprocess the embeddings
            vectors = scaler.fit_transform(vectors)

            if "pca" in embeddings_file:
                del embeddings_file["pca"]

            pca_h5_file = embeddings_file.create_group("pca")

            logger.info("Beginning PCA dimensionality reduction")
            # perform PCA dimensionality reduction
            pca_model = IncrementalPCA(n_components=pca_dim)

            pca_model, transformed, _ = train_fit_model(vectors, pca_model)
            # save both the reduced dimension vector and the full
            pca_h5_file["pca"] = transformed
            pca_h5_file["explained_variance"] = pca_model.explained_variance_ratio_

            logger.info("Saving the trained PCA model.")
            dump(pca_model, embeddings_save_loc / "pca_model.pkl")

            if compute_kmeans:
                logger.info("Performing K-means clustering on dataset")
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
                kmeans, _, labels = train_fit_model(pca_h5_file["pca"], kmeans)
                pca_h5_file["cluster_ids"] = labels
                dump(kmeans, embeddings_save_loc / f"kmeans_model.pkl")

            logger.info("Combining the models into a pipeline")

            if compute_kmeans:
                pipe = make_pipeline(scaler, pca_model, kmeans)
                pipeline_file = embeddings_save_loc / f"pca_pipeline_with_kmeans.pkl"
            else:
                pipe = make_pipeline(scaler, pca_model)
                pipeline_file = embeddings_save_loc / f"pca_pipeline_without_Kmeans.pkl"

            dump(pipe, pipeline_file)

        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

        finally:
            if client:
                client.close()


seed = 42

n_workers = 1
radius = 1
pca_dim = 70
n_clusters = 20
threads_per_worker = 2

embeddings_save_loc: pt = None
model_file: str = None
model = None
h5_file: pt = None
npy_file: pt = None
compute_kmeans = True
original_model = "mol2vec"
smi_to_vector = None


class Args:
    pca_dim: int = 70
    n_clusters: int = 20
    radius: int = 1
    embeddings_save_loc: str = None
    model_file: str = None
    npy_file: str = None
    embedding_pipeline_loc: str = None
    compute_kmeans: bool = False
    original_model: str = "mol2vec"
    PCA_pipeline_location: str = None


def main(args: Args):

    global original_model, smi_to_vector, pca_dim, n_clusters, radius, embeddings_save_loc, model, h5_file, npy_file, compute_kmeans

    pca_dim = args.pca_dim
    n_clusters = args.n_clusters
    radius = args.radius
    compute_kmeans = args.compute_kmeans

    logger.info(f"Computing kmeans: {compute_kmeans}")

    original_model = args.original_model
    embeddings_save_loc = pt(args.embeddings_save_loc) / original_model

    if not embeddings_save_loc.exists():
        embeddings_save_loc.mkdir(parents=True)

    logger.info(f"Embedding model: {smi_to_vec_dict}")
    smi_to_vector = smi_to_vec_dict[original_model]

    logger.info(f"Using model: {original_model} from {args.model_file}")

    if original_model == "mol2vec":
        model = load_model(args.model_file)
    else:
        model = load_model(args.model_file, use_joblib=True)
    h5_file = embeddings_save_loc / f"data.h5"
    npy_file = pt(args.npy_file)

    with ProgressBar():
        generate_embeddings()
