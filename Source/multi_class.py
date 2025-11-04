import os
import joblib
import logging
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(
    filename="C:/Users/User/Desktop/AI_Projects/Project_05/Log/multi_class.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelStrategy:
    def __init__(self):
        self.rf_base = RandomForestClassifier(random_state=42)
        self.dt_base = DecisionTreeClassifier(random_state=42)
        self.ovo_rf_model = None
        self.ovr_rf_model = None
        self.ovo_dt_model = None
        self.ovr_dt_model = None

    def train_ovo_random_forest(self, X_train, X_test, y_train, y_test, model_path, result_path):
        self.ovo_rf_model = OneVsOneClassifier(self.rf_base)
        self.ovo_rf_model.fit(X_train, y_train)
        preds = self.ovo_rf_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"OvO RandomForest accuracy: {acc:.3f}")
        self._save_results(self.ovo_rf_model, acc, model_path, result_path, "One-vs-One (Random Forest)")
        return acc

    def train_ovr_random_forest(self, X_train, X_test, y_train, y_test, model_path, result_path):
        self.ovr_rf_model = OneVsRestClassifier(self.rf_base)
        self.ovr_rf_model.fit(X_train, y_train)
        preds = self.ovr_rf_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"OvR RandomForest accuracy: {acc:.3f}")
        self._save_results(self.ovr_rf_model, acc, model_path, result_path, "One-vs-Rest (Random Forest)")
        return acc

    def train_ovo_decision_tree(self, X_train, X_test, y_train, y_test, model_path, result_path):
        self.ovo_dt_model = OneVsOneClassifier(self.dt_base)
        self.ovo_dt_model.fit(X_train, y_train)
        preds = self.ovo_dt_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"OvO DecisionTree accuracy: {acc:.3f}")
        self._save_results(self.ovo_dt_model, acc, model_path, result_path, "One-vs-One (Decision Tree)")
        return acc

    def train_ovr_decision_tree(self, X_train, X_test, y_train, y_test, model_path, result_path):
        self.ovr_dt_model = OneVsRestClassifier(self.dt_base)
        self.ovr_dt_model.fit(X_train, y_train)
        preds = self.ovr_dt_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"OvR DecisionTree accuracy: {acc:.3f}")
        self._save_results(self.ovr_dt_model, acc, model_path, result_path, "One-vs-Rest (Decision Tree)")
        return acc



    def _save_results(self, model, acc, model_path, result_path, model_name):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"{model_name} saqlandi: {model_path}")

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(f"{model_name} Accuracy: {acc:.3f}\n")

        print(f"[{model_name}] Accuracy: {acc:.3f} → {result_path}")
