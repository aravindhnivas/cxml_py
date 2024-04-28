import numpy as np
import h5py
from mol2vec import features
from pathlib import Path as pt
from umdalib.utils import load_model

# from multiprocessing import cpu_count
import numpy as np

from rdkit import Chem

# from gensim.models import word2vec
# from tqdm.auto import tqdm

USE_DASK = True
from dask import array as da
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


def smi_to_vector(smi: str, model, radius: int) -> list[np.ndarray]:
    """
    Given a model, convert a SMILES string into the corresponding
    NumPy vector.
    """
    # Molecule from SMILES will break on "bad" SMILES; this tries
    # to get around sanitization (which takes a while) if it can
    mol = Chem.MolFromSmiles(smi, sanitize=False)

    if not mol:
        return None

    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    # generate a sentence from rdkit molecule
    sentence = features.mol2alt_sentence(mol, radius)
    # generate vector embedding from sentence and model
    vector = features.sentences2vec([sentence], model)
    return vector


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


class EmbeddingModel(object):
    def __init__(self, w2vec_obj, transform=None, radius: int = 1) -> None:
        self._model = w2vec_obj
        self._transform = transform
        self._radius = radius
        self._covariance = None

    @property
    def model(self):
        return self._model

    @property
    def transform(self):
        return self._transform

    @property
    def radius(self):
        return self._radius

    def vectorize(self, smi: str):

        try:
            vector = smi_to_vector(smi, self.model, self.radius)

            if self._transform is not None:
                # the clustering is always the last step, which we ignore
                for step in self.transform.steps[: len(self.transform.steps) - 1]:
                    vector = step[1].transform(vector)
            return vector[0]
        except:
            return da.from_array(np.zeros(pca_dim))

    def __call__(self, smi: str):
        return self.vectorize(smi)

    @classmethod
    def from_pkl(cls, w2vec_path, transform_path=None, **kwargs):
        w2vec_obj = load_model(w2vec_path)
        if transform_path:
            transform_obj = load(transform_path)
        else:
            transform_obj = None
        return cls(w2vec_obj, transform_obj, **kwargs)

    def save(self, path: str):
        dump(self, path)
        logger.info(f"Saved model to {path}.")


def generate_embeddings():

    with h5py.File(h5_file, "w") as embeddings_file:

        np_vec = np.load(npy_file, allow_pickle=True)
        vectors = np.vstack(np_vec)
        embeddings_file["vectors"] = vectors

        if USE_DASK:
            client = Client(threads_per_worker=threads_per_worker, n_workers=n_workers)
            vectors = da.from_array(np.vstack(np_vec))

        logger.info(f"{vectors.shape=}")

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

        logger.info("Performing K-means clustering on dataset")
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans, _, labels = train_fit_model(pca_h5_file["pca"], kmeans)
        pca_h5_file["cluster_ids"] = labels
        dump(kmeans, embeddings_save_loc / "kmeans_model.pkl")

        logger.info("Combining the models into a pipeline")
        pipe = make_pipeline(scaler, pca_model, kmeans)
        dump(pipe, embeddings_save_loc / "embedding_pipeline.pkl")

        # generate a convenient wrapper for all the functionality

        embedder = EmbeddingModel(m2v_model, transform=pipe)
        embedder_file = embeddings_save_loc / "EmbeddingModel.pkl"

        dump(embedder, embedder_file)
        logger.info(f"Embedding model saved to disk ({embedder_file}). Exiting.")

    return embedder


seed = 42

n_workers = 1
radius = 1
pca_dim = 70
n_clusters = 20
threads_per_worker = 2

embeddings_save_loc: pt = None
model_file: str = None

h5_file: pt = None
npy_file: pt = None


class Args:
    pca_dim: int = 70
    n_clusters: int = 20
    radius: int = 1
    embeddings_save_loc: str = None
    model_file: str = None
    npy_file: str = None
    embedding_pipeline_loc: str = None


def main(args: Args):

    global pca_dim, n_clusters, radius, embeddings_save_loc, m2v_model, h5_file, npy_file

    pca_dim = args.pca_dim
    n_clusters = args.n_clusters
    radius = args.radius

    embeddings_save_loc = pt(args.embeddings_save_loc)
    m2v_model = load_model(args.model_file)
    h5_file = embeddings_save_loc / f"embeddings_PCA_{pca_dim}dim.h5"
    npy_file = pt(args.npy_file)

    if args.embedding_pipeline_loc:
        logger.info("Loading existing pipeline.")

        pipe = load(args.embedding_pipeline_loc)
        embedder = EmbeddingModel(m2v_model, transform=pipe)
        embedder_file = embeddings_save_loc / "mol2vec_pca.pkl"

        dump(embedder, embedder_file)
        logger.info(f"Embedding model saved to disk ({embedder_file}). Exiting.")

        return

    generate_embeddings()
