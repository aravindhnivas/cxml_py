from dataclasses import dataclass
from waitress import serve
from optuna.storages import RDBStorage
from optuna_dashboard import wsgi
from umdalib.logger import Paths


@dataclass
class Args:
    port: int = 8080


def main(args: Args):
    print("Starting Optuna dashboard...")
    print(f"Using port: {args.port}")

    filename = Paths().app_log_dir / "optuna/storage.db"
    if not filename.exists():
        raise FileNotFoundError(f"Database file not found: {filename}")

    db_url = f"sqlite:///{str(filename)}"
    print(f"Using database URL: {db_url}")

    storage = RDBStorage(db_url)
    app = wsgi(storage)

    print(f"Optuna dashboard is running on http://localhost:{args.port}")
    serve(app, port=args.port, url_scheme="http")


if __name__ == "__main__":
    main(Args())
