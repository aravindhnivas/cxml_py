from pathlib import Path as pt

base_loc1 = pt(
    "/Users/aravindhnivas/Documents/ML-properties/[PHYSICAL CONSTANTS OF ORGANIC COMPOUNDS]/"
)
base_loc2 = pt(
    "/Users/aravindhnivas/Documents/ML-properties/[CRITICAL CONSTANTS OF ORGANIC COMPOUNDS]/"
)

processed_data_dirs = [
    base_loc1
    / "tmp_C_processed_data/analysis_data/filtered/tmpC_topelements_processed_data",
    base_loc1
    / "tbp_C_processed_data/analysis_data/filtered/tbp_topelements_processed_data",
    base_loc1
    / "vp_kPa_25C_filtered_ydata_processed_data/analysis_data/filtered/vp_kPa_25C_topelements_processed_data",
    base_loc2 / "Pc_MPa_processed_data",
    base_loc2 / "Tc_K_processed_data",
]

plots_dir = pt("/Users/aravindhnivas/Documents/ML-properties/plots/")
titles = ["MP", "BP", "VP", "CP", "CT"]
total_counts = [7476, 4915, 398, 777, 819]

embeddings = ["mol2vec_embeddings", "VICGAE_embeddings"]
models = ["gbr", "catboost", "xgboost", "lgbm"]
models_labels = ["GBR", "CatBoost", "XGBoost", "LGBM"]
