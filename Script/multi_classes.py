import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source')))

from pipeline import AllPipeline
from multi_class import ModelStrategy
from sklearn.model_selection import train_test_split

data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Preprocessed_data', 'preprocessed_top_5.csv')
data_path = os.path.abspath(data_path)


pipe = AllPipeline(data_path=data_path, target_col="Category")
pipe.load_data()
pipe.preprocessing_data()

X, y = pipe.creating_pipeline()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


strategy = ModelStrategy()


strategy.train_ovo_random_forest(X_train, X_test, y_train, y_test,
    model_path="C:/Users/User/Desktop/AI_Projects/Project_05/Model/ovo_random_forest.joblib",
    result_path="C:/Users/User/Desktop/AI_Projects/Project_05/Result/ovo_random_forest.txt")

strategy.train_ovr_random_forest(X_train, X_test, y_train, y_test,
    model_path="C:/Users/User/Desktop/AI_Projects/Project_05/Model/ovr_random_forest.joblib",
    result_path="C:/Users/User/Desktop/AI_Projects/Project_05/Result/ovr_random_forest.txt")


strategy.train_ovo_decision_tree(X_train, X_test, y_train, y_test,
    model_path="C:/Users/User/Desktop/AI_Projects/Project_05/Model/ovo_decision_tree.joblib",
    result_path="C:/Users/User/Desktop/AI_Projects/Project_05/Result/ovo_decision_tree.txt")

strategy.train_ovr_decision_tree(X_train, X_test, y_train, y_test,
    model_path="C:/Users/User/Desktop/AI_Projects/Project_05/Model/ovr_decision_tree.joblib",
    result_path="C:/Users/User/Desktop/AI_Projects/Project_05/Result/ovr_decision_tree.txt")
