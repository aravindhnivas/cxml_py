from pathlib import Path as pt

user_loc = pt("/Users/aravindhnivas")
# root_loc = user_loc / "Documents/ML-properties"
root_loc = (
    user_loc
    / "Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/ML-properties"
)

base_loc1 = root_loc / "[PHYSICAL CONSTANTS OF ORGANIC COMPOUNDS]"
base_loc2 = root_loc / "[CRITICAL CONSTANTS OF ORGANIC COMPOUNDS]"


saved_states_1 = [
    "tmp_C_processed_data/analysis_data/filtered/tmpC_topelements_processed_data",
    "tbp_C_processed_data/analysis_data/filtered/tbp_topelements_processed_data",
    "vp_kPa_25C_filtered_ydata_processed_data/analysis_data/filtered/vp_kPa_25C_topelements_processed_data",
    
]
saved_states_2 = ["Pc_MPa_processed_data", "Tc_K_processed_data"]
saved_states = saved_states_1 + saved_states_2

processed_data_dirs = [base_loc1 / f for f in saved_states_1] + [base_loc2 / f for f in saved_states_2]
embedded_vectors_dir = [d / "embedded_vectors" for d in processed_data_dirs]

# processed_data_dirs = [
#     base_loc1
#     / "tmp_C_processed_data/analysis_data/filtered/tmpC_topelements_processed_data",
#     base_loc1
#     / "tbp_C_processed_data/analysis_data/filtered/tbp_topelements_processed_data",
#     base_loc1
#     / "vp_kPa_25C_filtered_ydata_processed_data/analysis_data/filtered/vp_kPa_25C_topelements_processed_data",
#     base_loc2 / "Pc_MPa_processed_data",
#     base_loc2 / "Tc_K_processed_data",
# ]

plots_dir = root_loc / "plots"
titles = ["MP", "BP", "VP", "CP", "CT"]
total_counts = [7476, 4915, 398, 777, 819]

embeddings_names = ["Mol2Vec", "VICGAE"]
embeddings_dirname = ["mol2vec_embeddings", "VICGAE_embeddings"]
models = ["gbr", "catboost", "xgboost", "lgbm"]
models_labels = ["GBR", "CatBoost", "XGBoost", "LGBM"]
