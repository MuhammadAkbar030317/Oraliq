import logging
from Source.models import Train

log_file = r"C:/Users/User/Desktop/AI_Projects/Project_05/Log/main_training.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    logging.info("Model train boshlandi")
    data_path = r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Engineered_data/log_transform.csv"
    trainer = Train(data_path=data_path)
    trainer.load_data()
    logging.info("Data muvaffaqiyatli yuklandi.")

    # Decision Tree 
    logging.info("Decision Tree modeli train qilinmoqda...")
    trainer.train_model(model_type="decision_tree", target_col="Category")
    logging.info("Decision Tree modeli train yakunlandi.")

    # Random Forest
    logging.info("Random Forest modeli train qilinmoqda...")
    trainer.train_model(model_type="random_forest", target_col="Category")
    logging.info("Random Forest modeli train yakunlandi.")

    logging.info("Barcha model training yakunlandi")

except Exception as e:
    logging.error(f"Trainingda xatolik: {e}")
    print(f"Xatolik: {e}")

