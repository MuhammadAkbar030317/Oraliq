import sys
sys.path.append(r"C:/Users/User/Desktop/AI_Projects/Project_05/Source")

from tuning import HyperparameterTuner

if __name__ == "__main__":
    data_path = r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Preprocessed_data/preprocessed_top_5.csv"
    target_col = "Category"  # maqsad ustuning nomi

    tuner = HyperparameterTuner(data_path=data_path, target_col=target_col)

    # Random Forest uchun tuning
    rf_model, rf_acc = tuner.tune_model(model_type="random_forest")

    # Decision Tree uchun tuning
    dt_model, dt_acc = tuner.tune_model(model_type="decision_tree")

    print(f"RandomForest Accuracy: {rf_acc:.3f}")
    print(f"DecisionTree Accuracy: {dt_acc:.3f}")
