from waitress import serve
from optuna.storages import RDBStorage
from optuna_dashboard import wsgi
from umdalib.utils import Paths


def main():
    filename = Paths().app_log_dir / "optuna/storage.db"
    if not filename.exists():
        raise FileNotFoundError(f"Database file not found: {filename}")

    db_url = f"sqlite:///{str(filename)}"
    print(f"Using database URL: {db_url}")

    storage = RDBStorage(db_url)
    app = wsgi(storage)

    return app


if __name__ == "__main__":
    print("Optuna dashboard is running on http://localhost:8080")
    app = main()
    serve(app, port=8080, url_scheme="http")
