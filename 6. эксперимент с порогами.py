
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv("facies_data.csv")
features = [
    "GR", "ILD_log10", "DeltaPHI",
    "PHIND", "PE", "RELPOS", "NM_M"
]
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

oof_p1 = np.zeros(len(dev))
cv = GroupKFold(n_splits=3)

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=10,
        class_weight=None,
        random_state=42
    )

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    probabilities = model.predict_proba(
        X.iloc[val_idx]
    )

    #находим столбец именно Facies 1
    column_1 = np.where(model.classes_ == 1)[0][0]

    oof_p1[val_idx] = probabilities[:, column_1]

    print(f"Fold {fold}: вероятности записаны")

# True = настоящий песчаник
# False = любая другая фация
is_sandstone = (y.to_numpy() == 1)

#чекаем пороги
thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]

print("\nПОРОГИ ОБНАРУЖЕНИЯ ПЕСЧАНИКА")
print(
    "threshold | precision | recall | "
    "detected | true_positive | false_positive"
)
for threshold in thresholds:

    # True, если оценка песчаника >= порога
    detected = (oof_p1 >= threshold)

    precision = precision_score(
        is_sandstone,
        detected,
        zero_division=0
    )
    recall = recall_score(
        is_sandstone,
        detected,
        zero_division=0
    )
    true_positive = np.sum(
        detected & is_sandstone
    )
    false_positive = np.sum(
        detected & ~is_sandstone
    )
    print(
        f"{threshold:9.2f} | "
        f"{precision:9.3f} | "
        f"{recall:6.3f} | "
        f"{detected.sum():8d} | "
        f"{true_positive:13d} | "
        f"{false_positive:14d}"
    )
