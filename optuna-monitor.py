from waitress import serve
from optuna.storages import RDBStorage
from optuna_dashboard import wsgi
from umdalib.utils import Paths

filename = Paths().app_log_dir / "optuna/storage.db"
storage = RDBStorage(f"sqlite:///{str(filename)}")
app = wsgi(storage)


if __name__ == "__main__":
    print("Optuna dashboard is running on http://localhost:8080")
    serve(app, port=8080, url_scheme="http")
