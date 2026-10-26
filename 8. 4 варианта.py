
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)
df = pd.read_csv("facies_data.csv")
log_features = [
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE"
]
base_features = log_features + [
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


sorted_dev = dev.sort_values(
    ["Well Name", "Depth"]
).copy()

window_size = 5

mean_features = []
std_features = []

for feature in log_features:

    mean_name = feature + "_mean_5"
    std_name = feature + "_std_5"

    sorted_dev[mean_name] = (
        sorted_dev.groupby("Well Name")[feature]
        .transform(
            lambda values: values.rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean()
        )
    )

    sorted_dev[std_name] = (
        sorted_dev.groupby("Well Name")[feature]
        .transform(
            lambda values: values.rolling(
                window=window_size,
                center=True,
                min_periods=2
            ).std()
        )
    )

    dev[mean_name] = sorted_dev[mean_name]
    dev[std_name] = sorted_dev[std_name]

    mean_features.append(mean_name)
    std_features.append(std_name)

new_features = mean_features + std_features

print("\nНОВЫЕ ПРИЗНАКИ")
print(new_features)

print("\nПРИМЕР НОВЫХ ПРИЗНАКОВ")

print(
    dev[
        [
            "Well Name",
            "Depth",
            "GR",
            "GR_mean_5",
            "GR_std_5"
        ]
    ].head(10).round(3).to_string(index=False)
)

print("\nПропуски в новых признаках:")
print(dev[new_features].isna().sum().to_string())

assert not dev[new_features].isna().any().any()

feature_sets = {
    "A_base": base_features,
    "B_mean": base_features + mean_features,
    "C_std": base_features + std_features,
    "D_mean_std": base_features + mean_features + std_features
}

cv = GroupKFold(n_splits=3)
folds = list(
    cv.split(
        dev,
        dev["Facies"],
        dev["Well Name"]
    )
)

results = []

for name, features in feature_sets.items():
    print("МОДЕЛЬ:", name)
    print("Количество признаков:", len(features))

    X = dev[features]
    y = dev["Facies"]

    fold_accuracy = []
    fold_f1 = []

    all_true = []
    all_pred = []

    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        train_wells = dev.iloc[train_idx]["Well Name"]
        val_wells = dev.iloc[val_idx]["Well Name"]

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

        accuracy = accuracy_score(y_val, predictions)

        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )

        fold_accuracy.append(accuracy)
        fold_f1.append(macro_f1)

        all_true.extend(y_val)
        all_pred.extend(predictions)

        print(
            f"Fold {fold_number}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}"
        )

    oof_accuracy = accuracy_score(all_true, all_pred)

    oof_macro_f1 = f1_score(
        all_true,
        all_pred,
        labels=classes,
        average="macro",
        zero_division=0
    )

    recalls = recall_score(
        all_true,
        all_pred,
        labels=classes,
        average=None,
        zero_division=0
    )

    results.append({
        "model": name,
        "mean_accuracy": sum(fold_accuracy) / len(fold_accuracy),
        "mean_macro_f1": sum(fold_f1) / len(fold_f1),
        "oof_accuracy": oof_accuracy,
        "oof_macro_f1": oof_macro_f1,
        **{
            f"recall_{facies}": recall
            for facies, recall in zip(classes, recalls)
        }
    })

summary = pd.DataFrame(results)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print(summary.round(4).to_string(index=False))
