
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)
df = pd.read_csv("../facies_data.csv")
base_features = [
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE"
]
feature_sets = {
    "A_base": base_features,
    "B_relpos": base_features + ["RELPOS"],
    "E_nm_only": base_features + ["NM_M"],
    "C_both": base_features + ["RELPOS", "NM_M"]
}
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

print("\nФАЦИИ ПО СКВАЖИНАМ")
print(
    pd.crosstab(
        dev["Well Name"],
        dev["Facies"]
    ).to_string()
)
print("\nСООТВЕТСТВИЕ NM_M И FACIES")
print(
    pd.crosstab(
        dev["Facies"],
        dev["NM_M"]
    ).to_string()
)
cv = GroupKFold(n_splits=3)
folds = list(
    cv.split(
        dev,
        dev["Facies"],
        dev["Well Name"]
    )
)
print("\nКОЛИЧЕСТВО ФАЦИЙ В TRAIN И VALIDATION")
for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
    train = dev.iloc[train_idx]
    val = dev.iloc[val_idx]
    print(f"\nFOLD {fold_number}")
    counts = pd.DataFrame({
        "train": train["Facies"].value_counts(),
        "validation": val["Facies"].value_counts()
    })
    counts = counts.reindex(classes).fillna(0).astype(int)
    print(counts.to_string())

weights = {
    "ordinary": None,
    "balanced": "balanced"
}
results = []

for feature_name, features in feature_sets.items():

    X = dev[features]
    y = dev["Facies"]

    for model_name, class_weight in weights.items():

        fold_f1 = []

        oof_pred = np.zeros(len(dev), dtype=int)

        for fold, (train_idx, val_idx) in enumerate(
            folds,
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
                class_weight=class_weight,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            oof_pred[val_idx] = predictions

            score = f1_score(
                y_val,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0
            )

            fold_f1.append(score)


        oof_accuracy = accuracy_score(y, oof_pred)

        oof_f1 = f1_score(
            y,
            oof_pred,
            labels=classes,
            average="macro",
            zero_division=0
        )

        recalls = recall_score(
            y,
            oof_pred,
            labels=classes,
            average=None,
            zero_division=0
        )

        results.append({
            "features": feature_name,
            "model": model_name,
            "mean_fold_f1": np.mean(fold_f1),
            "oof_accuracy": oof_accuracy,
            "oof_macro_f1": oof_f1,
            "recall_1": recalls[0],
            "recall_9": recalls[8]
        })

summary = pd.DataFrame(results)
print("СРАВНЕНИЕ ГЕОЛОГИЧЕСКОГО КОНТЕКСТА")
print(
    summary.round(4).to_string(index=False)
)