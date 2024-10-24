from distutils.sysconfig import get_python_lib
from pathlib import Path as pt

import umdalib

site_pkgs = pt(get_python_lib())
print(f"{site_pkgs=}")
distributed = site_pkgs / "distributed/distributed.yaml"
dask = site_pkgs / "dask/dask.yaml"

hiddenimports = [
    "umdalib.server",
    "umdalib.getVersion",
    "umdalib.training",
    "astrochem_embedding",
]
icons_dir = pt(umdalib.__file__).parent / "../icons"
icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]

# include templates folder in umdalib.server.flask
templates_dir = pt(umdalib.__file__).parent / "server/flask/templates"
templates_files = [
    (str(file.resolve()), "umdalib/server/flask/templates")
    for file in templates_dir.glob("*")
]

distributed_datas = [(str(distributed.resolve()), "distributed")]
dask_datas = [(str(dask.resolve()), "dask")]

distributed_http = site_pkgs / "distributed/http"
distributed_http_datas = [(str(distributed_http.resolve()), "distributed/http")]

libxgboost = site_pkgs / "xgboost/lib"
libxgboost_datas = [(str(libxgboost.resolve()), "xgboost/lib")]
xgboost_VERSION = site_pkgs / "xgboost/VERSION"
libxgboost_datas += [(str(xgboost_VERSION.resolve()), "xgboost")]

lgbm = site_pkgs / "lightgbm/lib"
lgbm_datas = [(str(lgbm.resolve()), "lightgbm/lib")]
lgbm_VERSION = site_pkgs / "lightgbm/VERSION.txt"
lgbm_datas += [(str(lgbm_VERSION.resolve()), "lightgbm")]

lightning_fabric_datas = []
lightning_fabric_version = site_pkgs / "lightning_fabric/version.info"
lightning_fabric_datas += [
    (str(lightning_fabric_version.resolve()), "lightning_fabric")
]

umdalib_version = pt(umdalib.__file__).parent / "__version__.dat"
version_datas = [(str(umdalib_version.resolve()), "umdalib")]

datas = (
    icons_files
    + templates_files
    + distributed_datas
    + dask_datas
    + distributed_http_datas
    + libxgboost_datas
    + lgbm_datas
    + lightning_fabric_datas
    + version_datas
)
