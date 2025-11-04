# import os
# import logging
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# log_dir = "C:/Users/User/Desktop/AI_Projects/Project_05/Log"
# os.makedirs(log_dir, exist_ok=True)

# logging.basicConfig(
#     filename=os.path.join(log_dir, "training.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# class Train:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.df = None
#         self.model = None

#     def load_data(self):
#         try:
#             self.df = pd.read_csv(self.data_path)
#             logging.info(f"Data yuklandi: {self.data_path}")
#         except Exception as e:
#             logging.error(f"Data yuklashda xatolik: {e}")

#     def train_model(self, model_type="decision_tree", target_col="Category"):
#         try:
#             logging.info(f"{model_type} modelini train qilish boshlandi.")
            
#             x = self.df.drop([target_col,"Market value"],axis=1)
#             y = self.df[target_col]

#             x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#             if model_type == "decision_tree":
#                 self.model = DecisionTreeClassifier(random_state=42)
#             elif model_type == "logistic_regression":
#                 self.model = LogisticRegression(max_iter=700, random_state=42)
#             elif model_type == "knn":
#                 self.model = KNeighborsClassifier(n_neighbors=5)
#             elif model_type == "random_forest":
#                 self.model = RandomForestClassifier(random_state=42)
#             else:
#                 print("model_type faqat 'decision_tree', 'logistic_regression', 'knn' yoki 'random_forest' bo'lishi kerak.")

#             self.model.fit(x_train, y_train)
#             y_pred = self.model.predict(x_test)
#             acc = accuracy_score(y_test, y_pred)
#             report = classification_report(y_test, y_pred)

#             logging.info(f"{model_type} training yakunlandi. Accuracy: {acc:.4f}")
#             print(f"\n{model_type} accuracy: {acc:.4f}")
#             print(report)

#         except Exception as e:
#             logging.error(f"{model_type} train jarayonida xatolik: {e}")


#            # Natijani listga qo‘shamiz
#             self.results.append({
#                 "Dataset": self.dataset_name,
#                 "Model": model_type,
#                 "Accuracy": acc
#             })

#             # Har safar natijani faylga yozamiz (append rejimida)
#             self.save_results(append=True)

#         except Exception as e:
#             logging.error(f"{model_type} train jarayonida xatolik: {e}")
#             print(f"{model_type} train jarayonida xatolik: {e}")

#     def save_results(self, append=True):
#         """Natijalarni Result/results.py fayliga yozadi (append bilan)."""
#         try:
#             result_dir = r"C:/Users/User/Desktop/AI_Projects/Project_05/Results"
#             os.makedirs(result_dir, exist_ok=True)
#             result_path = os.path.join(result_dir, "results.py")

#             # Agar fayl mavjud va append=True bo‘lsa, eski ma’lumotlarni o‘qiymiz
#             old_data = []
#             if append and os.path.exists(result_path):
#                 with open(result_path, "r", encoding="utf-8") as f:
#                     lines = f.readlines()
#                     for line in lines:
#                         if line.strip().startswith("{'Dataset'"):
#                             old_data.append(line)

#             # Faylni yangilaymiz
#             with open(result_path, "w", encoding="utf-8") as f:
#                 f.write("# Auto-generated model training results\n")
#                 f.write("results = [\n")
#                 for line in old_data:
#                     f.write(line)
#                 for r in self.results:
#                     f.write(
#                         f"    {{'Dataset': '{r['Dataset']}', 'Model': '{r['Model']}', 'Accuracy': {r['Accuracy']:.4f}}},\n"
#                     )
#                 f.write("]\n\n")
#                 f.write("def show_results():\n")
#                 f.write("    print('Model Results by Dataset (Accuracy)')\n")
#                 f.write("    print('-' * 50)\n")
#                 f.write("    for r in results:\n")
#                 f.write("        print(f\"{r['Dataset']:<25}{r['Model']:<20}{r['Accuracy']:.4f}\")\n")

#             logging.info(f"Natijalar '{result_path}' fayliga yozildi.")

#         except Exception as e:
#             logging.error(f"Natijalarni saqlashda xatolik: {e}")
#             print(f"Natijalarni saqlashda xatolik: {e}")




# ---------------------------------------
import os
import logging
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class Train:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.results = {}  # <---- shu joy yangi qo‘shildi!

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data yuklandi: {self.data_path}")
        except Exception as e:
            logging.error(f"Data yuklashda xatolik: {e}")

    def train_model(self, model_type="decision_tree", target_col="Category"):
        try:
            x = self.df.drop(columns=[target_col], errors="ignore")
            y = self.df[target_col]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

            # modelni tanlash
            if model_type == "decision_tree":
                self.model = DecisionTreeClassifier(random_state=42)
            elif model_type == "random_forest":
                self.model = RandomForestClassifier(random_state=42)
            elif model_type == "knn":
                self.model = KNeighborsClassifier()
            elif model_type == "logistic_regression":
                self.model = LogisticRegression(max_iter=2000)
            else:
                raise ValueError(f"Noma’lum model turi: {model_type}")

            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)

            # result dict ga saqlash
            self.results[model_type] = {
                "Accuracy": round(acc, 4),
                "Report": report
            }

            # log yozish
            logging.info(f"{model_type} training yakunlandi. Accuracy: {acc:.4f}")
            print(f"\n✅ {model_type} accuracy: {acc:.4f}")

            # resultlarni faylga yozish
            self._save_results(model_type, acc, report)

        except Exception as e:
            logging.error(f"{model_type} train jarayonida xatolik: {e}")

    def _save_results(self, model_type, acc, report):
        """Natijalarni Result/results.py ichiga yozish"""
        results_dir = r"C:/Users/User/Desktop/AI_Projects/Project_05/Result"
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "results.py")

        with open(result_file, "a", encoding="utf-8") as f:
            f.write(f"\n# Dataset: {os.path.basename(self.data_path)}")
            f.write(f"\n# Model: {model_type}")
            f.write(f"\nAccuracy = {acc:.4f}\n")
            f.write(report)
            f.write("\n" + "="*80 + "\n")
