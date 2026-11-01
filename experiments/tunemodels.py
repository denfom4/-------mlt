
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score

from catboost import CatBoostClassifier
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "facies_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "logo_6"
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
assert set(dev["Well Name"]).isdisjoint(
    set(df.iloc[holdout_idx]["Well Name"])
)
X = dev[features]
y = dev["Facies"]
groups = dev["Well Name"]

cv = LeaveOneGroupOut()
folds = list(cv.split(X, y, groups))

configurations = [
    (
        "RF_baseline",
        RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42
        )
    ),
    (
        "RF_depth8_leaf5",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )
    ),
    (
        "RF_depth10_leaf3",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42
        )
    ),
    (
        "CB_baseline",
        CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=2
        )
    ),
    (
        "CB_depth4",
        CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=2
        )
    ),
    (
        "CB_depth6",
        CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=2
        )
    ),
    (
        "CB_faster_learning",
        CatBoostClassifier(
            iterations=400,
            depth=5,
            learning_rate=0.10,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=2
        )
    )
]

results = []
well_results = []

for name, model_template in configurations:

    print("КОНФИГУРАЦИЯ:", name)

    oof_pred = np.full(len(dev), -1, dtype=int)
    fold_f1_scores = []

    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        well_name = groups.iloc[val_idx].unique()[0]

        model = clone(model_template)
        model.fit(X_train, y_train)

        predictions = np.asarray(
            model.predict(X_val)
        ).ravel().astype(int)
        oof_pred[val_idx] = predictions
        well_accuracy = accuracy_score(
            y_val,
            predictions
        )

        well_f1 = f1_score(
            y_val,
            predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )
        fold_f1_scores.append(well_f1)
        well_results.append({
            "model": name,
            "well": well_name,
            "accuracy": well_accuracy,
            "macro_f1": well_f1
        })
        print(
            f"{well_name}: "
            f"Accuracy={well_accuracy:.4f}, "
            f"Macro F1={well_f1:.4f}"
        )
    assert np.all(oof_pred != -1)
    accuracy = accuracy_score(y, oof_pred)
    macro_f1 = f1_score(
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
        "model": name,
        "oof_accuracy": accuracy,
        "oof_macro_f1": macro_f1,
        "mean_fold_f1": np.mean(fold_f1_scores),
        "min_fold_f1": np.min(fold_f1_scores)
    }
    for facies, recall in zip(classes, recalls):
        result[f"recall_{facies}"] = recall

    results.append(result)

summary = pd.DataFrame(results)
summary = summary.sort_values(
    "oof_macro_f1",
    ascending=False
)
well_table = pd.DataFrame(well_results)
print("РЕЗУЛЬТАТЫ ПОДБОРА")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    RESULTS_DIR / "tuning_comparison.csv",
    index=False
)
well_table.to_csv(
    RESULTS_DIR / "tuning_by_well.csv",
    index=False
)
print("\nРезультаты сохранены в:", RESULTS_DIR)
