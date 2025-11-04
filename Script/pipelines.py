import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source')))
from pipeline import AllPipeline

data_path = r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Raw_data/Top5_Leagues_2005_2025.csv"

pipe = AllPipeline(data_path=data_path, target_col="Category")
pipe.load_data()
pipe.preprocessing_data()
pipe.creating_pipeline()
model, acc = pipe.train_model()
pipe.save_results(
    result_save_path="../Result/pipeline_result.txt",
    accuracy=acc
)
