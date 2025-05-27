from distutils.sysconfig import get_python_lib
from pathlib import Path as pt

import cxml_lib

site_pkgs = pt(get_python_lib())
print(f"{site_pkgs=}")
distributed = site_pkgs / "distributed/distributed.yaml"
dask = site_pkgs / "dask/dask.yaml"

hiddenimports = [
    "cxml_lib.server",
    "cxml_lib.getVersion",
    "cxml_lib.molecule_analysis",
    "cxml_lib.load_file",
    "cxml_lib.vectorize_molecules",
    "cxml_lib.dimensionality_reduction",
    "cxml_lib.ml_training",
    "cxml_lib.start_redis_worker",
    "astrochem_embedding",
]

icons_dir = pt(cxml_lib.__file__).parent / "../icons"
icons_files = [(str(file.resolve()), "icons") for file in icons_dir.glob("*")]

# include templates folder in cxml_lib.server.flask
templates_dir = pt(cxml_lib.__file__).parent / "server/flask/templates"
templates_files = [
    (str(file.resolve()), "cxml_lib/server/flask/templates")
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

cxml_lib_version = pt(cxml_lib.__file__).parent / "__version__.dat"
version_datas = [(str(cxml_lib_version.resolve()), "cxml_lib")]

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
