import os
import logging
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

log_dir = "C:/Users/User/Desktop/AI_Projects/Project_05/Log"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Train:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data yuklandi: {self.data_path}")
        except Exception as e:
            logging.error(f"Data yuklashda xatolik: {e}")

    def train_model(self, model_type="decision_tree", target_col="Category"):
        try:
            logging.info(f"{model_type} modelini train qilish boshlandi.")
            
            x = self.df.drop(columns=[target_col,"Market Value"])
            y = self.df[target_col]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

            if model_type == "decision_tree":
                self.model = DecisionTreeClassifier(random_state=42)
            elif model_type == "random_forest":
                self.model = RandomForestClassifier(random_state=42)
            else:
                print("model_type faqat 'decision_tree' yoki 'random_forest' bo'lishi kerak.")

            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logging.info(f"{model_type} training yakunlandi. Accuracy: {acc:.4f}")
            print(f"\n{model_type} accuracy: {acc:.4f}")
            print(report)

        except Exception as e:
            logging.error(f"{model_type} train jarayonida xatolik: {e}")


