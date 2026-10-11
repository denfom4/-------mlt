import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("facies_data.csv")
features = ["GR","ILD_log10","DeltaPHI","PHIND","PE"]

X = df[features]
y = df["Facies"]
groups = df["Well Name"]

split_by_wells = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    split_by_wells.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]
###################################################################
cv = GroupKFold(n_splits=4)
depth_list = [
    3,
    5,
    7,
    10,
    15,
    20,
    None
]

for max_depth in depth_list:

    scores_rf = []

    for train_idx, val_idx in cv.split(
        X,
        y,
        groups
    ):

        X_train_cv = X.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]

        y_train_cv = y.iloc[train_idx]
        y_val_cv = y.iloc[val_idx]

        model_rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=max_depth,
            random_state=42
        )

        model_rf.fit(X_train_cv,y_train_cv)
        y_val_pred = model_rf.predict(X_val_cv)

        accuracy_rf = accuracy_score(y_val_cv,y_val_pred)
        scores_rf.append(accuracy_rf)

    mean_accuracy = (
        sum(scores_rf)
        / len(scores_rf)
    )

    print(
        f"max_depth={max_depth}: "
        f"accuracy={mean_accuracy:.4f}"
    )