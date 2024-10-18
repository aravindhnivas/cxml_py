from typing import TypedDict, Literal, Optional, List, Dict, Union, Tuple


# Define the structure of the inner dictionaries
class DataEntry(TypedDict):
    y_true: List[float]
    y_pred: List[float]
    y_linear_fit: List[float]


class DataType(TypedDict):
    test: DataEntry
    train: DataEntry


# Define the Embedding type
Embedding = Literal["mol2vec", "VICGAE"]


# Define the MLStats TypedDict
class MLStats(TypedDict):
    r2: float
    mse: float
    rmse: float
    mae: float


# Define the CVScores TypedDict
class CVScores(TypedDict):
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    scores: List[float]


# Define the CV_scoring_methods type
CV_scoring_methods = Literal["r2", "mse", "rmse", "mae"]

# Define the LearningCurveData type
LearningCurveData = Dict[str, Dict[Literal["test", "train"], CVScores]]


class LearningCurve(TypedDict):
    data: LearningCurveData
    train_sizes: List[float]
    sizes: Tuple[float, float, int]
    CV: int
    scoring: Literal["r2"]


# Define the CVScoresData type
CVScoresData = Dict[Literal["test", "train"], Dict[CV_scoring_methods, CVScores]]


# Define the PlotData and Layout TypedDicts (assuming simplified structures)
class PlotData(TypedDict, total=False):
    x: List[float]
    y: List[float]
    type: str
    name: str


class Layout(TypedDict, total=False):
    title: str
    xaxis: Dict[str, Union[str, int, float]]
    yaxis: Dict[str, Union[str, int, float]]


# Define the MLResults TypedDict
class MLResults(TypedDict):
    learning_curve_plotly_data: Optional[Dict[str, Union[List[PlotData], Layout]]]
    embedding: Embedding
    PCA: bool
    data_shapes: Dict[str, List[int]]
    train_stats: MLStats
    test_stats: MLStats
    model: str
    bootstrap: bool
    bootstrap_nsamples: Optional[int]
    cross_validation: bool
    cv_fold: Optional[int]
    cv_scores: Optional[CVScoresData]
    best_params: Optional[Dict[str, Union[str, int, bool, None]]]
    best_score: Optional[float]
    timestamp: str
    time: str
