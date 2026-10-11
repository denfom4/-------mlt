import pandas as pd
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("facies_data.csv")
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


#GroupKFold
cv = GroupKFold(n_splits=4)
model_params = {
    "n_estimators": 50,
    "max_depth": 5,
    "random_state": 42
}
# Списки для результатов
accuracy_scores = []
balanced_accuracy_scores = []
macro_f1_scores = []

all_y_true = []
all_y_pred = []
# кросс-валидация
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):
    X_train_cv = X.iloc[train_idx]
    X_val_cv = X.iloc[val_idx]

    y_train_cv = y.iloc[train_idx]
    y_val_cv = y.iloc[val_idx]

    model_rf = RandomForestClassifier(**model_params)

    model_rf.fit(X_train_cv,y_train_cv)

    y_val_pred = model_rf.predict(
        X_val_cv
    )
    # оценка
    accuracy = accuracy_score(
        y_val_cv,
        y_val_pred
    )
    balanced_accuracy = balanced_accuracy_score(
        y_val_cv,
        y_val_pred
    )
    macro_f1 = f1_score(
        y_val_cv,
        y_val_pred,
        average="macro"
    )
    # сохраняем результаты
    accuracy_scores.append(accuracy)
    balanced_accuracy_scores.append(
        balanced_accuracy
    )
    macro_f1_scores.append(macro_f1)


    # сохраняем предсказания
    all_y_true.extend(y_val_cv)
    all_y_pred.extend(y_val_pred)


    print(
        f"Fold {fold}: "
        f"accuracy={accuracy:.4f}, "
        f"balanced_accuracy={balanced_accuracy:.4f}, "
        f"macro_f1={macro_f1:.4f}"
    )

# срзнач
print("\nСредние значения:")
print(
    f"Accuracy: "
    f"{sum(accuracy_scores) / len(accuracy_scores):.4f}"
)
print(
    f"Balanced accuracy: "
    f"{sum(balanced_accuracy_scores) / len(balanced_accuracy_scores):.4f}"
)
print(
    f"Macro F1: "
    f"{sum(macro_f1_scores) / len(macro_f1_scores):.4f}"
)
# classification report
print("\nClassification report:")
print(
    classification_report(
        all_y_true,
        all_y_pred
    )
)
# cm
cm = confusion_matrix(
    all_y_true,
    all_y_pred
)
print("\nConfusion matrix:")
print(cm)