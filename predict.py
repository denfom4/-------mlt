
from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_depth4.cbm"
INFO_PATH = PROJECT_ROOT / "models" / "catboost_depth4_info.json"

# для первого теста возьмём существующий CSV
# позже заменим этот путь на файл новой скважины
#в new_well.csv не нужны фации !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
INPUT_PATH = PROJECT_ROOT / "data" / "new_well.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INFO_PATH, "r", encoding="utf-8") as file:
    model_info = json.load(file)
features = model_info["features"]
train_wells = set(model_info["train_wells"])


model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))
print("Модель загружена:", MODEL_PATH)


df = pd.read_csv(INPUT_PATH)
# в первом тесте используем скважины разработки,чтобы проверить работу программы
#!!!!не независимая оценка качества модели

if "Well Name" not in df.columns:
    raise ValueError("Отсутствует столбец Well Name")
#исправил для той херни с сайтом, не забыть о df = df[df["Well Name"].isin(train_wells)].copy()
if df.empty:
    raise ValueError("Не найдено строк для тестового запуска")
print("Количество интервалов:", len(df))


missing_columns = [
    feature
    for feature in features
    if feature not in df.columns
]
if missing_columns:
    raise ValueError(
        f"В данных отсутствуют признаки: {missing_columns}"
    )
X_new = df[features]
if X_new.isna().any().any():
    raise ValueError(
        "В признаках обнаружены пропуски NaN"
    )


predictions = np.asarray(
    model.predict(X_new)
).ravel().astype(int)
probabilities = model.predict_proba(X_new)


result = df.copy()
result["Predicted_Facies"] = predictions
result["Max_Probability"] = probabilities.max(axis=1)

# сохраняем вероятности каждой фации
# порядок столбцов берём у самой модели
for column_number, facies in enumerate(model.classes_):
    result[f"Probability_Facies_{int(facies)}"] = (
        probabilities[:, column_number]
    )

output_path = OUTPUT_DIR / "facies_predictions.csv"
result.to_csv(
    output_path,
    index=False
)
print("\nПример результатов:")
print(
    result[
        ["Well Name", "Depth",
         "Predicted_Facies", "Max_Probability"]
    ].head(10).round(3).to_string(index=False)
)
print("\nРезультат сохранён:", output_path)
