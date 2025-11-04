
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class AllPipeline:
    def __init__(self, data_path, target_col):
        self.data_path = data_path
        self.target_col = target_col
        self.data = None
        self.pipeline = None

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Fayl topilmadi: {self.data_path}")
        self.data = pd.read_csv(self.data_path)

        if "Market value" in self.data.columns:
            self.data["Market value"] = (
                self.data["Market value"]
                .astype(str)
                .replace('[\$,]', '', regex=True)
                .replace('nan', None)
            )
            self.data["Market value"] = pd.to_numeric(self.data["Market value"], errors="coerce")

        if self.data["Market value"].isna().all():
            raise ValueError("'Market value' ustunida hech qanday son yo‘q!")

        if 'Category' not in self.data.columns:
            self.data['Category'] = pd.qcut(
                self.data['Market value'], q=3, labels=[2, 1, 0]
            )

        logging.info(f"Data yuklandi: {self.data.shape}")
        return self.data

    def preprocessing_data(self):
        self.data.dropna(inplace=True)
        logging.info("NaN qiymatlar o‘chirildi.")
        return self.data

    def creating_pipeline(self):
        x = self.data.drop([self.target_col, "Market value"], axis=1)
        y = self.data[self.target_col]

        categorical_cols = x.select_dtypes(include=["object"]).columns
        numeric_cols = x.select_dtypes(exclude=["object"]).columns

        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ])

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=150, random_state=42))
        ])

        logging.info("Pipeline yaratildi.")
        return x, y

    def train_model(self):
        x, y = self.creating_pipeline()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline.fit(x_train, y_train)
        logging.info("Model train qilindi.")

        preds = self.pipeline.predict(x_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"Accuracy score: {acc:.3f}")
        print(f"Accuracy score: {acc:.3f}")

        return self.pipeline, acc

    def save_results(self, result_save_path, accuracy):
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
        with open(result_save_path, "a", encoding="utf-8") as f:
            f.write(f"Accuracy: {accuracy:.3f}\n")
        logging.info(f"Aniqlik natijasi saqlandi: {result_save_path}")
        print(f"Accuracy natijasi yozildi → {result_save_path}")
