from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all template files from rq_dashboard
datas = collect_data_files(
    "rq_dashboard", include_py_files=True, includes=["**/*.html", "**/*.js", "**/*.css"]
)

# Make sure we collect all submodules
hiddenimports = collect_submodules("rq_dashboard")

# Add additional RQ dependencies
hiddenimports.extend(["rq", "redis", "flask", "jinja2"])
