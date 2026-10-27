
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
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
holdout = df.iloc[holdout_idx].copy()
print("Скважины разработки:")
print(list(dev["Well Name"].unique()))
print("\nОтложенные скважины:")
print(list(holdout["Well Name"].unique()))
assert set(dev["Well Name"]).isdisjoint(
    set(holdout["Well Name"])
)
X = dev[features]
y = dev["Facies"]
groups = dev["Well Name"]

cv = GroupKFold(n_splits=3)
oof_pred = np.zeros(len(dev), dtype=int)

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    train_wells = groups.iloc[train_idx]
    val_wells = groups.iloc[val_idx]

    assert set(train_wells).isdisjoint(set(val_wells))

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=10,
        class_weight=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    oof_pred[val_idx] = predictions

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0
    )

    print(f"\nFold {fold}")
    print("Validation wells:", list(val_wells.unique()))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

print("ИТОГОВЫЕ OOF-МЕТРИКИ")
print(
    "Accuracy:",
    round(accuracy_score(y, oof_pred), 4)
)
print(
    "Macro F1:",
    round(
        f1_score(
            y,
            oof_pred,
            labels=classes,
            average="macro",
            zero_division=0
        ),
        4
    )
)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y,
        oof_pred,
        labels=classes,
        zero_division=0
    )
)
print("\nCONFUSION MATRIX")
print(
    confusion_matrix(
        y,
        oof_pred,
        labels=classes
    )
)
