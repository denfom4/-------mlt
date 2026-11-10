
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "catboost_depth4.cbm"
INFO_PATH = PROJECT_ROOT / "models" / "catboost_depth4_info.json"

# Файл для запуска без дополнительных параметров
DEFAULT_INPUT = PROJECT_ROOT / "data" / "new_well.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "predictions"

def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Не найдена модель: {MODEL_PATH}"
        )

    if not INFO_PATH.is_file():
        raise FileNotFoundError(
            f"Не найден файл параметров: {INFO_PATH}"
        )

    with open(INFO_PATH, "r", encoding="utf-8") as file:
        info = json.load(file)

    features = info["features"]

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    # Проверяем соответствие модели и JSON
    model_features = list(model.feature_names_)

    if model_features and model_features != features:
        raise ValueError(
            "Порядок признаков модели и JSON не совпадает"
        )

    return model, features

def validate_data(df, features, default_well_name):
    df = df.copy()

    if df.empty:
        raise ValueError("Входной CSV пуст")

    # Depth нужен для построения каротажной панели
    required_columns = ["Depth"] + features
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Отсутствуют обязательные столбцы: "
            f"{missing_columns}"
        )
    # Если названия скважины нет, используем имя файла
    if "Well Name" not in df.columns:
        df["Well Name"] = default_well_name
        print(
            "Столбца Well Name нет. "
            f"Используем имя: {default_well_name}"
        )
    if (
        df["Well Name"].isna().any()
        or df["Well Name"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "В столбце Well Name есть пустые значения"
        )

    # Проверяем глубины и все признаки модели
    numeric_columns = ["Depth"] + features
    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )
        invalid = ~np.isfinite(
            values.to_numpy(dtype=float)
        )
        if invalid.any():
            row_numbers = (
                np.flatnonzero(invalid)[:5] + 2
            ).tolist()
            raise ValueError(
                f"В столбце '{column}' есть пустые "
                "или некорректные значения. "
                f"Строки CSV: {row_numbers}"
            )
        df[column] = values
    # Повторяющиеся глубины не удаляем!
    duplicate_count = df.duplicated(
        ["Well Name", "Depth"]
    ).sum()
    if duplicate_count > 0:
        print(
            f"Предупреждение: {duplicate_count} строк "
            "имеют повторяющуюся глубину. "
            "Они будут сохранены."
        )
    return df

def predict_facies(
    df,
    model,
    features,
    default_well_name="Unknown_well"
):
    df = validate_data(
        df,
        features,
        default_well_name
    )
    # Строго соблюдаем порядок признаков
    X = df[features]
    # Истинная Facies НЕ используется
    predictions = np.asarray(
        model.predict(X)
    ).ravel().astype(int)
    probabilities = model.predict_proba(X)
    result = df.copy()
    # Идентификатор исходной строки
    # Нужен в том числе при одинаковых глубинах
    if "Source_Row_ID" not in result.columns:
        result.insert(
            0,
            "Source_Row_ID",
            np.arange(1, len(result) + 1)
        )
    result["Predicted_Facies"] = predictions

    result["Max_Probability"] = (
        probabilities.max(axis=1)
    )
    # Номера классов определяем по самой модели
    for index, facies in enumerate(model.classes_):

        column_name = (
            f"Probability_Facies_{int(facies)}"
        )
        result[column_name] = (
            probabilities[:, index]
        )
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Классификация литофаций по данным ГИС"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Путь к CSV с данными скважины"
    )
    args = parser.parse_args()
    input_path = Path(args.csv).expanduser()
    # Относительные пути считаем от корня проекта
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.is_file():
        raise FileNotFoundError(
            f"CSV не найден: {input_path}"
        )
    print("Загружаем модель...")
    model, features = load_model()
    print("Загружаем данные...")
    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )
    print("Классифицируем литофации...")
    result = predict_facies(
        df,
        model,
        features,
        default_well_name=input_path.stem
    )
    # Создаём папку при необходимости
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    output_path = (
        OUTPUT_DIR /
        f"{input_path.stem}_predictions.csv"
    )
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )
    print("\nКЛАССИФИКАЦИЯ ЗАВЕРШЕНА")
    print("Интервалов:", len(result))
    print(
        "Скважин:",
        result["Well Name"].nunique()
    )
    print("\nПервые предсказания:")
    print(
        result[
            [
                "Well Name",
                "Depth",
                "Predicted_Facies",
                "Max_Probability"
            ]
        ].head(10).round(3).to_string(index=False)
    )
    print("\nРезультат сохранён:")
    print(output_path)


if __name__ == "__main__":
    main()
