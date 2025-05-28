# ChemXploreML Python Backend

This repository contains the Python backend for the ChemXploreML desktop application, which implements the machine learning framework described in the paper: [Machine Learning Pipeline for Molecular Property Prediction Using ChemXploreML](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00516).

Please visit the [Documentation](https://aravindhnivas.github.io/ChemXploreML-docs/) to download the desktop application. To access the desktop application source code, please visit the [ChemXploreML](https://github.com/aravindhnivas/ChemXploreML) repository.

## Overview

ChemXploreML is a powerful machine learning framework designed for chemical space exploration and molecular property prediction. This Python backend provides the core functionality for:

- Molecular feature generation and representation
- Machine learning model training and evaluation
- Chemical space visualization
- Property prediction and uncertainty estimation
- Model interpretation and explainability

## Features

- **Advanced ML Algorithms**: Support for XGBoost, LightGBM, CatBoost, and scikit-learn models
- **Chemical Space Analysis**: Integration with PCA, UMAP, t-SNE, KernelPCA, PHATE, ISOMAP, LaplacianEigenmaps, TriMap and FactorAnalysis for dimensionality reduction
- **Model Optimization**: Hyperparameter tuning with Optuna
- **Task Queue**: Asynchronous processing with Redis and RQ
- **Data Quality**: Integration with CleanLab for data quality assessment
- **Deep Learning**: Support for transformer-based models and custom neural networks (soon to be added)

## Requirements

- [Rye](https://rye.astral.sh/) package manager

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aravindhnivas/cxml_py.git
cd cxml_py
```

2. Ensure you have Rye installed and create and activate a virtual environment:

```bash
rye sync

# for unix/macOS
source .venv/bin/activate

# or for windows
.venv\Scripts\activate
```

## Usage

- Start the desktop application ChemXploreML.
- Navigate to the 'Settings' tab to start the server.

## Project Structure

```
cxml_py/
├── src/
│   └── cxml_lib/        # Core library code
├── pyproject.toml      # Project configuration
├── requirements.lock   # Locked dependencies
└── README.md           # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

Marimuthu, A. N.; McGuire, B. A. Machine Learning Pipeline for Molecular Property Prediction Using ChemXploreML. J. Chem. Inf. Model. 2025. <https://doi.org/10.1021/acs.jcim.5c00516>.

## Support

For support, please open an issue in the GitHub repository or contact <aravindhnivas28@gmail.com>.

## Acknowledgments

- [Kelvin Lee's UMDA repository](https://github.com/laserkelvin/umda) for the mol2vec implementation.
- [Kelvin Lee's astrochem_embedding repository](https://github.com/laserkelvin/astrochem_embedding) for the VICGAE implementation.
- The ML pipeline is inspired by K. Lee's [Machine Learning of Interstellar Chemical Inventories](https://iopscience.iop.org/article/10.3847/2041-8213/ac194b) paper.

I would like to thank the authors and maintainers of the following libraries for their invaluable contributions:

### Core Scientific Computing

- [NumPy](https://numpy.org/) - Array computing and linear algebra
- [SciPy](https://scipy.org/) - Scientific computing and optimization
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [Dask](https://dask.org/) - Parallel computing and task scheduling

### Machine Learning

- [Scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
- [XGBoost](https://xgboost.readthedocs.io/en/stable/) - Gradient boosting framework
- [LightGBM](https://lightgbm.readthedocs.io/en/stable/) - Light gradient boosting machine
- [CatBoost](https://catboost.ai/) - Gradient boosting on decision trees
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [SHAP](https://shap.readthedocs.io/en/latest/) - Model interpretability
- [CleanLab](https://docs.cleanlab.ai/stable/index.html) - Data quality and label error detection

### Deep Learning

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) - Deep learning training framework
- [Transformers](https://huggingface.co/docs/transformers/index) - State-of-the-art NLP

### Chemical Informatics

- [RDKit](https://www.rdkit.org/) - Cheminformatics and machine learning
- [SELFIES](https://github.com/aspuru-guzik-group/selfies) - String-based molecular representation

### Visualization

- [Matplotlib](https://matplotlib.org/) - Plotting library
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization
- [PHATE](https://phate.readthedocs.io/en/stable/) - Dimensionality reduction
- [UMAP](https://umap-learn.readthedocs.io/en/latest/) - Uniform Manifold Approximation
- [TriMap](https://github.com/eamid/trimap) - Dimensionality reduction

### Web and API

- [Flask](https://flask.palletsprojects.com/en/stable/) - Web framework
- [Redis](https://redis.io/) - In-memory data store
- [RQ](https://rq.readthedocs.io/en/stable/) - Task queue
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/en/latest/) - WebSocket support

### Development Tools

- [Rye](https://rye.astral.sh/) - Python package manager
- [PyInstaller](https://pyinstaller.org/) - Application packaging
