import os
import sys
sys.path.append(r"C:/Users/User/Desktop/AI_Projects/Project_05/Source")
from preprocessing import Preprocessor

data_dir = r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Raw_data"
save_dir = r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Preprocessed"

datasets = [
    "log_transform.csv",
    "embedded_DT.csv",
    "filtered.csv",
    "wrapped_d_tree.csv",
    "wrapped_random_forest.csv"
]

for data_file in datasets:
    data_path = os.path.join(data_dir, data_file)
    pre = Preprocessor(data_path, save_dir)
    save_name = os.path.splitext(data_file)[0] + "_processed"
    pre.run_pipeline(save_name)
