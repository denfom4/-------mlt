
from pathlib import Path
import json

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "facies_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "catboost_depth4.cbm"
INFO_PATH = MODELS_DIR / "catboost_depth4_info.json"

features = [
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE",
    "RELPOS",
    "NM_M"
]

df = pd.read_csv(DATA_PATH)
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
dev_idx, holdout_idx = next(
    splitter.split(
        df,
        df["Facies"],
        df["Well Name"]
    )
)
dev = df.iloc[dev_idx].copy()
holdout = df.iloc[holdout_idx].copy()

train_wells = sorted(dev["Well Name"].unique().tolist())
holdout_wells = sorted(holdout["Well Name"].unique().tolist())

assert set(train_wells).isdisjoint(set(holdout_wells))

X_train = dev[features]
y_train = dev["Facies"]

assert not X_train.isna().any().any()
print("Скважины обучения:", train_wells)
print("Количество интервалов:", len(dev))
print("Отложенные скважины:", holdout_wells)

model = CatBoostClassifier(
    iterations=500,
    depth=4,
    learning_rate=0.05,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=2
)

model.fit(X_train, y_train)
print("\nОбучение завершено.")

model.save_model(str(MODEL_PATH))
print("Модель сохранена:", MODEL_PATH)


model_info = {
    "algorithm": "CatBoostClassifier",
    "model_file": MODEL_PATH.name,
    "features": features,
    "target": "Facies",
    "train_wells": train_wells,
    "excluded_wells": holdout_wells,
    "training_rows": len(dev),
    "parameters": {
        "iterations": 500,
        "depth": 4,
        "learning_rate": 0.05,
        "loss_function": "MultiClass",
        "random_seed": 42
    },
    "validation_method": "LeaveOneGroupOut",
    "note": (
        "Model selected using development-well "
        "cross-validation. Holdout not used in this training."
    )
}
with open(INFO_PATH, "w", encoding="utf-8") as file:
    json.dump(
        model_info,
        file,
        ensure_ascii=False,
        indent=4
    )

print("Информация о модели сохранена:", INFO_PATH)

#проверка что можно загрузить
loaded_model = CatBoostClassifier()
loaded_model.load_model(str(MODEL_PATH))
original_pred = model.predict(X_train.iloc[:10])
loaded_pred = loaded_model.predict(X_train.iloc[:10])
assert (original_pred == loaded_pred).all()
print("\nПроверка загрузки пройдена успешно!")
