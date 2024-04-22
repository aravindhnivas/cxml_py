import umdalib
from pathlib import Path as pt
from distutils.sysconfig import get_python_lib

site_pkgs = pt(get_python_lib())
print(f"{site_pkgs=}")
distributed = site_pkgs / "distributed/distributed.yaml"
dask = site_pkgs / "dask/dask.yaml"

hiddenimports = [
    "umdalib.server",
    "umdalib.getVersion",
    "umdalib.training",
    "umdalib.mol2vec",
    "umdalib.pca",
    "astrochem_embedding",
]
icons_dir = pt(umdalib.__file__).parent / "../icons"
icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]

distributed_datas = [(str(distributed.resolve()), "distributed")]
dask_datas = [(str(dask.resolve()), "dask")]

datas = icons_files + distributed_datas + dask_datas

# to include wandb and wandb_vendor (wandb_gql) in the final package
# pyarrow_vendored = site_pkgs / "pyarrow/vendored"
# pyarrow_vendored_datas = [(str(pyarrow_vendored.resolve()), "pyarrow/vendored")]
# datas = icons_files + distributed_datas + dask_datas + pyarrow_vendored_datas
