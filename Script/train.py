
# import logging
# import os
# import sys
# sys.path.append(r"C:/Users/User/Desktop/AI_Projects/Project_05/Source")
# from models import Train

# log_file = r"C:/Users/User/Desktop/AI_Projects/Project_05/Log/main_training.log"

# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# try:
#     logging.info("Model train boshlandi")
#     data_path =[ r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/log_transform.csv"
#     r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/embedded_DT.csv"
#     r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/filtered.csv"
#     r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/wrapped_d_tree.csv"
#     r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/wrapped_random_forest.csv"
#     ]
#     for path in data_path:
#         dataset_name = os.path.basename(path)
#         print(f"\n Training boshlanmoqda: {dataset_name}")
#         logging.info(f"{dataset_name} uchun training boshlandi")
#         trainer = Train(data_path=path)
#         trainer.load_data()
#         logging.info("Data  yuklandi.")

#         # Decision Tree 
#         logging.info("Decision Tree train.")
#         trainer.train_model(model_type="decision_tree", target_col="Category")
#         logging.info("Decision Tree train yakunlandi.")

#         # Logistic regression
#         logging.info("Logistic regression")
#         trainer.train_model(model_type="logistic_regression", target_col="Category")
#         logging.info("Logistic regression train yakunlandi.")

#         # KNN
#         logging.info("KNN train")
#         trainer.train_model(model_type="knn",target_col="Category")
#         logging.info("KNN train yakunlandi.")
#         # Random Forest
#         logging.info("Random Forest train")
#         trainer.train_model(model_type="random_forest", target_col="Category")
#         logging.info("Random Forest train yakunlandi.")

#         logging.info("Barcha model training yakunlandi")

# except Exception as e:
#     logging.error(f"Trainingda xatolik: {e}")
#     print(f"Xatolik: {e}")





# ---------------------------------------------


import logging
import os
import sys

sys.path.append(r"C:/Users/User/Desktop/AI_Projects/Project_05/Source")
from models import Train

log_file = r"C:/Users/User/Desktop/AI_Projects/Project_05/Log/main_training.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

data_paths = [
    r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/log_transform.csv",
    r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/embedded_DT.csv",
    r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/filtered.csv",
    r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/wrapped_d_tree.csv",
    r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/wrapped_random_forest.csv"
]

try:
    for path in data_paths:
        logging.info(f"Data yuklanmoqda: {path}")
        trainer = Train(data_path=path)
        trainer.load_data()

        for model in ["decision_tree", "random_forest", "knn", "logistic_regression", "xgboost"]:
            logging.info(f"{os.path.basename(path)} uchun {model} train boshlandi.")
            trainer.train_model(model_type=model, target_col="Category")
            logging.info(f"{os.path.basename(path)} uchun {model} train tugadi.")

    logging.info("✅ Barcha model training jarayonlari yakunlandi.")

except Exception as e:
    logging.error(f"Trainingda xatolik: {e}")
    print(f"Xatolik: {e}")
