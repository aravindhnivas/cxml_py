import umdalib
from pathlib import Path as pt
from distutils.sysconfig import get_python_lib

site_pkgs = pt(get_python_lib())
distributed = site_pkgs / "distributed/distributed.yaml"
dask = site_pkgs / "dask/dask.yaml"
print(f"{distributed=}, {dask=}")

hiddenimports = [
    "umdalib.server",
    "umdalib.getVersion",
    "umdalib.training",
    "wandb",
    # "wandb_gql",
]
icons_dir = pt(umdalib.__file__).parent / "../icons"
icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]

distributed_datas = [(str(distributed.resolve()), "distributed")]
dask_datas = [(str(dask.resolve()), "dask")]

datas = icons_files + distributed_datas + dask_datas
