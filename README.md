# ChemXploreML Python Backend

This repository contains the Python backend for the ChemXploreML desktop application, which implements the machine learning framework described in the paper: [Machine Learning Pipeline for Molecular Property Prediction Using ChemXploreML](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00516).

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
├── scripts/             # Utility scripts
├── tests/              # Test suite
├── pyproject.toml      # Project configuration
├── requirements.lock   # Locked dependencies
└── README.md          # This file
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
