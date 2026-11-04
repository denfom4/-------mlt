
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRUTH_PATH = PROJECT_ROOT / "data" / "facies_data.csv"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "predictions"
    / "facies_predictions.csv"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WELL_NAME = "CROSS H CATTLE"
classes = list(range(1, 10))
keys = ["Well Name", "Depth"]

#данные из исходного датасета по скв
truth = pd.read_csv(TRUTH_PATH)

truth = truth[
    truth["Well Name"] == WELL_NAME
][keys + ["Facies"]].copy()

truth = truth.rename(
    columns={"Facies": "True_Facies"}
)

#загрузка предикта
predictions = pd.read_csv(PREDICTIONS_PATH)

required = keys + ["Predicted_Facies"]

missing = [
    column
    for column in required
    if column not in predictions.columns
]

if missing:
    raise ValueError(
        f"В файле прогнозов нет столбцов: {missing}"
    )

predictions = predictions[
    predictions["Well Name"] == WELL_NAME
].copy()

# оставляем только нужные для оценки столбцы
columns = required.copy()
if "Max_Probability" in predictions.columns:
    columns.append("Max_Probability")

predictions = predictions[columns]
# проверка данных
if truth.empty:
    raise ValueError(
        f"Нет истинных фаций для {WELL_NAME}"
    )

if predictions.empty:
    raise ValueError(
        f"Нет предсказаний для {WELL_NAME}"
    )

# Каждая глубина внутри скважины должна быть уникальной
duplicates = truth[
    truth.duplicated(keys, keep=False)
]

print("\nСтроки с повторяющимися глубинами:")
print(duplicates.head(20).to_string(index=False))
print("Всего таких строк:", len(duplicates))

# Проверяем количество измерений
print("Строк в исходной разметке:", len(truth))
print("Строк с предсказаниями:", len(predictions))

if len(truth) != len(predictions):
    raise ValueError(
        "Количество исходных и предсказанных интервалов не совпадает"
    )

# Нумеруем повторения внутри каждой глубины испр
truth["_occurrence"] = (
    truth.groupby(keys).cumcount()
)

predictions["_occurrence"] = (
    predictions.groupby(keys).cumcount()
)
# сопоставление скв с глубиной испр
merge_keys = [
    "Well Name",
    "Depth",
    "_occurrence"
]

comparison = predictions.merge(
    truth,
    on=merge_keys,
    how="outer",
    validate="one_to_one",
    indicator=True
)

unmatched = comparison[
    comparison["_merge"] != "both"
]

if not unmatched.empty:
    print("\nСтроки без соответствия:")
    print(unmatched.head(10).to_string(index=False))

    raise ValueError(
        "Не все интервалы удалось сопоставить по глубине"
    )

comparison = comparison.drop(
    columns=["_merge"]
)

comparison = comparison.sort_values(
    "Depth"
).reset_index(drop=True)

comparison["Correct"] = (
    comparison["True_Facies"]
    == comparison["Predicted_Facies"]
)

print("\nСкважина:", WELL_NAME)
print("Сопоставлено интервалов:", len(comparison))


y_true = comparison["True_Facies"]
y_pred = comparison["Predicted_Facies"]

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=classes,
    average="macro",
    zero_division=0
)

supports = (
    y_true.value_counts()
    .reindex(classes, fill_value=0)
)

present_classes = [
    facies
    for facies in classes
    if supports[facies] > 0
]

macro_f1_present = f1_score(
    y_true,
    y_pred,
    labels=present_classes,
    average="macro",
    zero_division=0
)

recalls = recall_score(
    y_true,
    y_pred,
    labels=classes,
    average=None,
    zero_division=0
)

print("\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 (9 классов): {macro_f1:.4f}")
print(
    f"Macro F1 (классы скважины): "
    f"{macro_f1_present:.4f}"
)
report = classification_report(
    y_true,
    y_pred,
    labels=classes,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

comparison.to_csv(
    RESULTS_DIR / "matched_predictions.csv",
    index=False
)
metrics = pd.DataFrame([{
    "well": WELL_NAME,
    "intervals": len(comparison),
    "accuracy": accuracy,
    "macro_f1_9": macro_f1,
    "macro_f1_present": macro_f1_present
}])
metrics.to_csv(
    RESULTS_DIR / "metrics.csv",
    index=False
)
recall_table = pd.DataFrame({
    "Facies": classes,
    "Intervals": supports.to_numpy(),
    "Recall": recalls
})
recall_table.to_csv(
    RESULTS_DIR / "recall_by_facies.csv",
    index=False
)
print("\nRECALL ПО ФАЦИЯМ")
print(recall_table.round(3).to_string(index=False))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=classes
)

fig, ax = plt.subplots(figsize=(9, 7))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)
display.plot(
    ax=ax,
    cmap="Greens",
    values_format="d",
    colorbar=True
)
ax.set_title(f"{WELL_NAME} — Confusion Matrix")
plt.tight_layout()
fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=170
)
plt.close(fig)

#НОРМАЛИЗОВАННАЯ МАТРИЦА ОШИБОК
cm_normalized = confusion_matrix(
    y_true,
    y_pred,
    labels=classes,
    normalize="true"
)
fig, ax = plt.subplots(figsize=(9, 7))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=classes
)
display.plot(
    ax=ax,
    cmap="Greens",
    values_format=".2f",
    colorbar=True
)

ax.set_title(
    f"{WELL_NAME} — Normalized Confusion Matrix"
)
plt.tight_layout()
fig.savefig(
    RESULTS_DIR / "confusion_matrix_normalized.png",
    dpi=170
)
plt.close(fig)


# Тёмный цвет — классы с 40+ интервалами в скважине
colors = [
    "#166534" if supports[facies] >= 40
    else "#86BFA0"
    for facies in classes
]

labels = [
    f"Facies {facies} (n={supports[facies]})"
    for facies in classes
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    labels,
    recalls,
    color=colors
)
ax.invert_yaxis()
ax.set_xlim(0, 1.10)
ax.set_xlabel("Recall")
ax.set_title(f"{WELL_NAME} — Recall by Facies")

for bar, recall, support in zip(
    bars,
    recalls,
    supports.to_numpy()
):
    label = (
        f"{recall:.2f}" if support > 0
        else "нет примеров"
    )
    ax.text(
        recall + 0.015,
        bar.get_y() + bar.get_height() / 2,
        label,
        va="center"
    )
plt.tight_layout()
fig.savefig(
    RESULTS_DIR / "recall_by_facies.png",
    dpi=170
)
plt.close(fig)

print("\nОценка завершена.")
print("Результаты сохранены в:", RESULTS_DIR)
