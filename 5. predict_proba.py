
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
df = pd.read_csv("facies_data.csv")
features = [
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE",
    "RELPOS",
    "NM_M"
]
classes = list(range(1, 10))

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

assert set(dev["Well Name"]).isdisjoint(
    set(df.iloc[holdout_idx]["Well Name"])
)

X = dev[features]
y = dev["Facies"]
groups = dev["Well Name"]
cv = GroupKFold(n_splits=3)

oof_predictions = np.zeros(len(dev), dtype=int)

oof_probabilities = np.zeros(
    (len(dev), len(classes))
)

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=10,
        class_weight=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)

    print(f"\nFold {fold}")
    print("Classes:", model.classes_)
    print("Probabilities shape:", probabilities.shape)

    oof_predictions[val_idx] = predictions

    for column, facies in enumerate(model.classes_):
        oof_probabilities[val_idx, facies - 1] = (
            probabilities[:, column]
        )

print("ОБЩИЕ OOF-МЕТРИКИ")
print(
    "Accuracy:",
    round(accuracy_score(y, oof_predictions), 4)
)
print(
    "Macro F1:",
    round(
        f1_score(
            y,
            oof_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        ),
        4
    )
)
true_facies_1 = (y.to_numpy() == 1)

facies_1_predictions = oof_predictions[true_facies_1]

facies_1_probabilities = oof_probabilities[true_facies_1]


top2_indices = np.argsort(
    facies_1_probabilities,
    axis=1
)[:, -2:]

top2_classes = top2_indices + 1

facies_1_in_top2 = np.any(
    top2_classes == 1,
    axis=1
)

print("АНАЛИЗ FACIES 1 — SANDSTONE")
print(
    "Настоящих интервалов Facies 1:",
    true_facies_1.sum()
)
print(
    "Правильно распознано:",
    np.sum(facies_1_predictions == 1)
)
print(
    "Facies 1 входит в топ-2:",
    np.sum(facies_1_in_top2)
)
print(
    "Доля Facies 1 в топ-2:",
    round(facies_1_in_top2.mean(), 4)
)
print(
    "Средняя оценка вероятности Facies 1:",
    round(facies_1_probabilities[:, 0].mean(), 4)
)
print(
    "Максимальная оценка вероятности Facies 1:",
    round(facies_1_probabilities[:, 0].max(), 4)
)
print("\nВо что модель превращает настоящие Facies 1:")
print(
    pd.Series(facies_1_predictions)
    .value_counts()
    .sort_index()
    .to_string()
)


result_table = pd.DataFrame(
    facies_1_probabilities,
    columns=[f"P_Facies_{c}" for c in classes]
)
result_table.insert(
    0,
    "Predicted_Facies",
    facies_1_predictions
)
result_table.insert(
    0,
    "Well",
    groups.to_numpy()[true_facies_1]
)
print("\nПЕРВЫЕ 10 НАСТОЯЩИХ ИНТЕРВАЛОВ FACIES 1")
print(
    result_table.head(10).round(3).to_string(index=False)
)
