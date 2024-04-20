from mol2vec import features
from pathlib import Path as pt

import numpy as np

from rdkit import Chem
from gensim.models import word2vec

import h5py
from tqdm.auto import tqdm

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


def gen_corpus(
    in_file: str,
    out_file: str,
):

    if isinstance(in_file, pt):
        in_file = str(in_file.resolve())

    if isinstance(out_file, pt):
        out_file = str(out_file.resolve())

    features.generate_corpus(
        in_file, out_file, r=radius, sentence_type=sentence_type, n_jobs=n_workers
    )


def gen_model(
    in_file: str,
    out_file: str,
):

    if isinstance(in_file, pt):
        in_file = str(in_file.resolve())

    if isinstance(out_file, pt):
        out_file = str(out_file.resolve())

    model = features.train_word2vec_model(
        infile_name=in_file,
        outfile_name=out_file,
        vector_size=vector_size,
        min_count=min_count,
        n_jobs=n_workers,
    )

    return model


def smi_to_vector(smi: str, model) -> list[np.ndarray]:
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


def load_model(filepath: str):

    if isinstance(filepath, pt):
        filepath = str(filepath.resolve())

    return word2vec.Word2Vec.load(filepath)


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
        vector = smi_to_vector(smi, self.model, self.radius)
        #
        if self._transform is not None:
            # the clustering is always the last step, which we ignore
            for step in self.transform.steps[: len(self.transform.steps) - 1]:
                vector = step[1].transform(vector)
        return vector[0]

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


def stream_embeddings_to_hdf5(
    filename: str, all_smiles: list[str], model: word2vec.Word2Vec
):
    """
    Given a list of SMILES strings, and a model, stream the embeddings
    to an HDF5 file.

    This is useful for when you have a large number of SMILES strings that
    you want to embed, but don't want to load them all into memory at once.

    This function will create a new HDF5 file if one doesn't exist, and
    will append to an existing HDF5 file if one does exist.

    The HDF5 file will have two datasets:
    - `smiles`: the SMILES strings
    - `vectors`: the embeddings for each SMILES string

    The datasets will be stored in a group called `full_dim_{vector_size}`,
    where `vector_size` is the size of the embedding vectors.

    Parameters
    ----------
    filename : str
        The filename of the HDF5 file to write to
    all_smiles : list[str]
        A list of SMILES strings to embed
    model : word2vec.Word2Vec
        The Mol2Vec model to use for embedding
    radius : int, optional
        The radius to use for generating sentences, by default 1

    """

    with h5py.File(filename, "a") as h5_ref:
        for key in ["smiles", "vectors"]:
            try:
                del h5_ref[key]
            except KeyError:
                pass

        group_name = f"full_dim_{vector_size}"
        if group_name in h5_ref:
            del h5_ref[group_name]

        full_dim_subgroup = h5_ref.create_group(group_name)

        smiles_dataset = full_dim_subgroup.create_dataset(
            "smiles", (len(all_smiles),), dtype=h5py.string_dtype()
        )

        # vectorize all the SMILES strings, and then store it in the HDF5 array
        vectors = np.vstack(
            [smi_to_vector(smi, model, radius) for smi in tqdm(all_smiles)]
        )
        vector_dataset = full_dim_subgroup.create_dataset(
            "vectors", (len(vectors), vector_size), dtype="float32"
        )

        for index, smi, vec in zip(range(len(all_smiles)), all_smiles, vectors):

            if not smi:
                continue

            if vec is None:
                continue

            smiles_dataset[index] = smi
            vector_dataset[index] = vec


def generate_embeddings():

    with h5py.File(h5_target, "a") as embeddings_file:
        # output_file = h5py.File(output_target, "a")

        if USE_DASK:
            client = Client(threads_per_worker=2, n_workers=n_workers)
            vectors = da.from_array(embeddings_file["full_dim_300/vectors"])
        else:
            vectors = embeddings_file["full_dim_300/vectors"][:]

        print(f"{vectors.shape=}")

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

        pca_model, transformed, _ = train_fit_model(vectors, pca_model, n_workers)
        # save both the reduced dimension vector and the full
        pca_h5_file["pca"] = transformed
        pca_h5_file["explained_variance"] = pca_model.explained_variance_ratio_

        logger.info("Saving the trained PCA model.")
        dump(pca_model, embeddings_save_loc / "pca_model.pkl")

        logger.info("Performing K-means clustering on dataset")
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans, _, labels = train_fit_model(pca_h5_file["pca"], kmeans, n_workers)
        pca_h5_file["cluster_ids"] = labels
        dump(kmeans, embeddings_save_loc / "kmeans_model.pkl")

        logger.info("Combining the models into a pipeline")
        pipe = make_pipeline(scaler, pca_model, kmeans)
        dump(pipe, embeddings_save_loc / "embedding_pipeline.pkl")

        # generate a convenient wrapper for all the functionality
        embedder = EmbeddingModel(m2v_model, transform=pipe)
        dump(embedder, embeddings_save_loc / "EmbeddingModel.pkl")

        logger.info("Embedding model saved to disk. Exiting.")

    return embedder


seed = 42
n_clusters = 20  # Number of clusters to end up with from KMeans
n_workers = 32  # Number of cpu cores used for calculation
threads_per_worker = 2  # Number of threads per worker
min_count: int = (
    1  # Number of occurrences a word should have to be considered in training
)

vector_size = 300  # Size of the embedding vector
pca_dim = 70  # Number of dimensions to reduce to using PCA

m2v_model = None

h5_target = "embeddings.h5"
embeddings_save_loc = pt("embeddings")
embeddings_save_loc.mkdir(exist_ok=True)

radius: int = 1
sentence_type: str = (
    "alt"  # Method to use in model training. Options cbow and skip-gram, default: skip-gram
)


class Args:
    location: str
    filename: str
    radius: int
    sentence_type: str
    n_workers: int
    n_clusters: int
    vector_size: int
    pca_dim: int
    threads_per_worker: int
    min_count: int


def main(args: Args):
    global m2v_model, radius, sentence_type, n_workers, vector_size, min_count, n_clusters, pca_dim, threads_per_worker

    if args.radius:
        radius = args.radius
    if args.sentence_type:
        sentence_type = args.sentence_type
    if args.n_workers:
        n_workers = args.n_workers
    if args.vector_size:
        vector_size = args.vector_size
    if args.min_count:
        min_count = args.min_count
    if args.n_clusters:
        n_clusters = args.n_clusters
    if args.pca_dim:
        pca_dim = args.pca_dim
    if args.threads_per_worker:
        threads_per_worker = args.threads_per_worker

    m2v_model = gen_model(args.location, "m2v_model.pkl")
    gen_corpus(args.location, "corpus.txt")
    generate_embeddings()
