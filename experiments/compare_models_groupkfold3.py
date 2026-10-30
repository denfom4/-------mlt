
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from catboost_exp import CatBoostClassifier
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "facies_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "groupkfold3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA_PATH)
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

assert set(dev["Well Name"]).isdisjoint(
    set(holdout["Well Name"])
)

print("Скважины разработки:")
print(list(dev["Well Name"].unique()))

print("\nОтложенные скважины:")
print(list(holdout["Well Name"].unique()))

X = dev[features]
y = dev["Facies"]
groups = dev["Well Name"]

cv = GroupKFold(n_splits=3)
folds = list(cv.split(X, y, groups))

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=10,
        class_weight=None,
        random_state=42
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=10,
        class_weight=None,
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=2
    )
}
results = []
fold_results = []

prediction_table = dev[
    ["Well Name", "Depth", "Facies"]
].copy()
for model_name, model_template in models.items():
    print("МОДЕЛЬ:", model_name)
    oof_pred = np.zeros(len(dev), dtype=int)

    accuracies = []
    macro_f1_scores = []

    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        train_wells = groups.iloc[train_idx]
        val_wells = groups.iloc[val_idx]

        assert set(train_wells).isdisjoint(set(val_wells))

        model = clone(model_template)
        model.fit(X_train, y_train)
        predictions = np.asarray(
            model.predict(X_val)
        ).ravel().astype(int)

        oof_pred[val_idx] = predictions

        accuracy = accuracy_score(y_val, predictions)

        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )

        accuracies.append(accuracy)
        macro_f1_scores.append(macro_f1)

        print(
            f"Fold {fold_number}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}"
        )

        print(
            "Validation wells:",
            list(val_wells.unique())
        )

        fold_results.append({
            "model": model_name,
            "fold": fold_number,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "validation_wells": ", ".join(val_wells.unique())
        })

    oof_accuracy = accuracy_score(y, oof_pred)
    oof_macro_f1 = f1_score(
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
    result = {
        "model": model_name,
        "mean_accuracy": np.mean(accuracies),
        "mean_macro_f1": np.mean(macro_f1_scores),
        "oof_accuracy": oof_accuracy,
        "oof_macro_f1": oof_macro_f1
    }
    for facies, recall in zip(classes, recalls):
        result[f"recall_{facies}"] = recall

    results.append(result)
    prediction_table[model_name] = oof_pred

    print(f"\nOOF Accuracy: {oof_accuracy:.4f}")
    print(f"OOF Macro F1: {oof_macro_f1:.4f}")
    print("\nRecall по фациям:")

    for facies, recall in zip(classes, recalls):
        print(f"Facies {facies}: {recall:.3f}")

    cm = confusion_matrix(
        y,
        oof_pred,
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
    ax.set_title(f"{model_name} — OOF Confusion Matrix")
    plt.tight_layout()
    fig.savefig(
        RESULTS_DIR / f"{model_name}_confusion_matrix.png",
        dpi=150
    )
    plt.close(fig)

summary = pd.DataFrame(results)
summary = summary.sort_values(
    by="oof_macro_f1",
    ascending=False
)
fold_table = pd.DataFrame(fold_results)

print("СРАВНЕНИЕ ПО FOLDS")
print(fold_table.round(4).to_string(index=False))
print("ИТОГОВОЕ СРАВНЕНИЕ")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)
fold_table.to_csv(
    RESULTS_DIR / "fold_comparison.csv",
    index=False
)
prediction_table.to_csv(
    RESULTS_DIR / "oof_predictions.csv",
    index=False
)
print("\nГотово. Результаты сохранены в:")
print(RESULTS_DIR)
