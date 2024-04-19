import umdalib
from pathlib import Path as pt
from distutils.sysconfig import get_python_lib

site_pkgs = pt(get_python_lib())
distributed = site_pkgs / "distributed/distributed.yaml"
dask = site_pkgs / "dask/dask.yaml"

hiddenimports = [
    "umdalib.server",
    "umdalib.getVersion",
    "umdalib.training",
]
icons_dir = pt(umdalib.__file__).parent / "../icons"
icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]

distributed_datas = [(str(distributed.resolve()), "distributed")]
dask_datas = [(str(dask.resolve()), "dask")]

# to include wandb and wandb_vendor (wandb_gql) in the final package
wandb_vendor = site_pkgs / "wandb/vendor"
wandb_vendor_datas = [(str(wandb_vendor.resolve()), "wandb/vendor")]
datas = icons_files + distributed_datas + dask_datas + wandb_vendor_datas

# datas = icons_files + distributed_datas + dask_datas
