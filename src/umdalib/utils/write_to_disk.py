import h5py
import numpy as np
from tqdm.auto import tqdm


def stream_embeddings_to_hdf5(
    filename: str, all_smiles: list[str], model, vector_size: int, radius: int = 1
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
