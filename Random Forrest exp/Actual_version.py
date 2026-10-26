
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report
)
df = pd.read_csv("../facies_data.csv")
features = [
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE"
]
X = df[features]
y = df["Facies"]
groups = df["Well Name"]
classes = list(range(1, 10))

# отделяю скважины для разработки
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
dev_idx, holdout_idx = next(
    splitter.split(X, y, groups)
)
X_dev = X.iloc[dev_idx]
y_dev = y.iloc[dev_idx]
groups_dev = groups.iloc[dev_idx]

holdout_wells = groups.iloc[holdout_idx].unique()

print("Скважины для разработки:")
print(groups_dev.unique())

print("\nОтложенные скважины:")
print(holdout_wells)

# проверка отсутствия пересечения
assert set(groups_dev).isdisjoint(
    set(groups.iloc[holdout_idx])
)
#кросс-валид
cv = GroupKFold(n_splits=3)

#дефолт vs с подобранными весами
configurations = {
    "ordinary": None,
    "balanced": "balanced"
}

for name, weight in configurations.items():

    print("\n" + "=" * 50)
    print("Модель:", name)

    fold_scores = []

    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_dev, y_dev, groups_dev),
        start=1
    ):

        X_train = X_dev.iloc[train_idx]
        X_val = X_dev.iloc[val_idx]

        y_train = y_dev.iloc[train_idx]
        y_val = y_dev.iloc[val_idx]

        # Проверяем скважины
        train_wells = groups_dev.iloc[train_idx]
        val_wells = groups_dev.iloc[val_idx]

        assert set(train_wells).isdisjoint(set(val_wells))

        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=10,
            class_weight=weight,
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

        # Средний recall по фиксированным 9 классам.
        # Отсутствующий класс получает 0.
        macro_recall = recall_score(
            y_val,
            predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )

        fold_scores.append(macro_f1)

        all_true.extend(y_val)
        all_pred.extend(predictions)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.3f}, "
            f"Macro F1={macro_f1:.3f}, "
            f"Macro Recall={macro_recall:.3f}"
        )

        print("Validation wells:", list(val_wells.unique()))

    print("\nMean fold Macro F1:", sum(fold_scores) / len(fold_scores))

    print("\nОбщий OOF classification report:")
    print(
        classification_report(
            all_true,
            all_pred,
            labels=classes,
            zero_division=0
        )
    )
