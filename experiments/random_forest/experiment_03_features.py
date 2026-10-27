from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "facies_data.csv"
df = pd.read_csv(DATA_PATH)
base_features = [
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE"
]
feature_sets = {
    "A_base": base_features,

    "B_relpos": base_features + [
        "RELPOS"
    ],

    "C_relpos_nm": base_features + [
        "RELPOS",
        "NM_M"
    ],

    "D_all": base_features + [
        "RELPOS",
        "NM_M",
        "Depth"
    ]
}
y = df["Facies"]
groups = df["Well Name"]
classes = list(range(1, 10))

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
dev_idx, holdout_idx = next(
    splitter.split(df, y, groups)
)
dev_df = df.iloc[dev_idx].copy()
print("Скважины разработки:")
print(list(dev_df["Well Name"].unique()))

print("\nОтложенные скважины:")
print(list(df.iloc[holdout_idx]["Well Name"].unique()))

assert set(dev_df["Well Name"]).isdisjoint(
    set(df.iloc[holdout_idx]["Well Name"])
)
cv = GroupKFold(n_splits=3)

folds = list(
    cv.split(
        dev_df,
        dev_df["Facies"],
        dev_df["Well Name"]
    )
)
# RF
weights = {
    "ordinary": None,
    "balanced": "balanced"
}
results = []
# ====================================================
#перебор наборов признако
for set_name, features in feature_sets.items():
    print("ПРИЗНАКИ:", set_name)
    print(features)

    X = dev_df[features]
    y_dev = dev_df["Facies"]

    for model_name, class_weight in weights.items():

        print("\nМодель:", model_name)

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

            y_train = y_dev.iloc[train_idx]
            y_val = y_dev.iloc[val_idx]

            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=10,
                class_weight=class_weight,
                random_state=42
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            accuracy = accuracy_score(
                y_val,
                predictions
            )
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
                f"accuracy={accuracy:.4f}, "
                f"macro_f1={macro_f1:.4f}"
            )
        # срзнач по folds
        mean_accuracy = sum(fold_accuracy) / len(fold_accuracy)
        mean_f1 = sum(fold_f1) / len(fold_f1)
        # общие OOF метрики
        oof_accuracy = accuracy_score(
            all_true,
            all_pred
        )
        oof_f1 = f1_score(
            all_true,
            all_pred,
            labels=classes,
            average="macro",
            zero_division=0
        )
        # recall для каждой фации
        recalls = recall_score(
            all_true,
            all_pred,
            labels=classes,
            average=None,
            zero_division=0
        )
        print(f"Mean fold Accuracy: {mean_accuracy:.4f}")
        print(f"Mean fold Macro F1: {mean_f1:.4f}")
        print(f"OOF Accuracy: {oof_accuracy:.4f}")
        print(f"OOF Macro F1: {oof_f1:.4f}")
        print("Recall по фациям:")

        for facies, recall in zip(classes, recalls):
            print(f"  Facies {facies}: {recall:.3f}")
        results.append({
            "features": set_name,
            "model": model_name,
            "mean_accuracy": mean_accuracy,
            "mean_macro_f1": mean_f1,
            "oof_accuracy": oof_accuracy,
            "oof_macro_f1": oof_f1,
            **{f"recall_facies_{facies}": recall
                for facies, recall in zip(classes, recalls)}
        })


# итоги
summary = pd.DataFrame(results)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print(
    summary.round(4).to_string(index=False)
)
