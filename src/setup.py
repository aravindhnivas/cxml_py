from setuptools import find_packages, setup

setup(
    name="umdalib",
    packages=find_packages(),
    package_data={"umdalib": ["icons/*"]},
    install_requires=[
        "numpy",
        "scipy",
        "uncertainties",
        "matplotlib",
        "PyQt6",
        "flask_cors",
        "waitress",
    ],
    version="0.0.2",
    description="umdalib: a Python backend for UMDA_UI",
    author="Aravindh Nivas Marimuthu (Mcguire's group, MIT)",
    license="MIT",
)
