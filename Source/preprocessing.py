import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Log fayllar uchun papka
log_dir = r"C:/Users/User/Desktop/AI_Projects/Project_05/Log"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Preprocessor:
    def __init__(self, raw_path):
        """
        raw_path: Raw data faylning to‘liq manzili (CSV)
        """
        self.raw_path = raw_path
        self.df = None

        # Saqlash joyi
        self.prep_dir = r"C:/Users/User/Desktop/AI_Projects/Project_05/Data/Preprocessed"
        os.makedirs(self.prep_dir, exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.raw_path)
            logging.info(f"✅ Data yuklandi: {self.raw_path}")
        except Exception as e:
            logging.error(f"Data yuklashda xatolik: {e}")
            raise e

    def clean_data(self):
        """NaN va dublikatlarni olib tashlaydi"""
        try:
            self.df.drop_duplicates(inplace=True)
            self.df.dropna(inplace=True)
            logging.info("🧹 NaN va dublikatlar o‘chirildi.")
        except Exception as e:
            logging.error(f"Data cleaningda xatolik: {e}")

    def encode_features(self):
        """Kategorik ustunlarni LabelEncoder orqali raqamlaydi"""
        try:
            le = LabelEncoder()
            for col in self.df.select_dtypes(include=['object']).columns:
                self.df[col] = le.fit_transform(self.df[col])
            logging.info("🔢 Label encoding bajarildi.")
        except Exception as e:
            logging.error(f"Encodingda xatolik: {e}")

    def scale_data(self):
        """Raqamli ustunlarni normallashtiradi"""
        try:
            scaler = StandardScaler()
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
            logging.info("📏 Scaling bajarildi.")
        except Exception as e:
            logging.error(f"Scalingda xatolik: {e}")

    def save_processed(self):
        """Natijani Preprocessed papkaga saqlaydi"""
        try:
            file_name = os.path.splitext(os.path.basename(self.raw_path))[0]
            save_path = os.path.join(self.prep_dir, f"{file_name}_processed.csv")
            self.df.to_csv(save_path, index=False)
            logging.info(f"💾 Fayl saqlandi: {save_path}")
            print(f"✅ Preprocessed fayl: {save_path}")
        except Exception as e:
            logging.error(f"Saqlashda xatolik: {e}")

    def run_pipeline(self):
        """To‘liq preprocessing pipeline"""
        self.load_data()
        self.clean_data()
        self.encode_features()
        self.scale_data()
        self.save_processed()
