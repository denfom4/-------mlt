
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
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
facies_names = {
    1: "Nonmarine sandstone",
    2: "Coarse siltstone",
    3: "Fine siltstone",
    4: "Marine siltstone/shale",
    5: "Mudstone",
    6: "Wackestone",
    7: "Dolomite",
    8: "Packstone-grainstone",
    9: "Phylloid-algal bafflestone"
}
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

    print(
        f"Fold {fold}: "
        f"Accuracy={accuracy:.4f}, "
        f"Macro F1={macro_f1:.4f}"
    )

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
#просто цветную cm добавлю
cm = confusion_matrix(
    y,
    oof_pred,
    labels=classes
)
fig, ax = plt.subplots(figsize=(10, 8))
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
ax.set_title("Random Forest — OOF Confusion Matrix")
ax.set_xlabel("Predicted Facies")
ax.set_ylabel("True Facies")
plt.tight_layout()
plt.show()

recalls = recall_score(
    y,
    oof_pred,
    labels=classes,
    average=None,
    zero_division=0
)

supports = (
    y.value_counts()
    .reindex(classes, fill_value=0)
)

frequent_facies = [
    facies
    for facies in classes
    if supports[facies] >= 400
]
print("\nЧастые фации:", frequent_facies)
print("\nRECALL ПО ВСЕМ ФАЦИЯМ")
for facies, recall in zip(classes, recalls):
    print(
        f"Facies {facies}: "
        f"{recall:.3f} | "
        f"{facies_names[facies]} | "
        f"интервалов: {supports[facies]}"
    )

labels = [
    f"{facies}: {facies_names[facies]}"
    for facies in classes
]
colors = [
    "forestgreen" if facies in frequent_facies
    else "mediumseagreen"
    for facies in classes
]
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(
    labels,
    recalls,
    color=colors
)
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel("Recall")
ax.set_title("Random Forest — OOF Recall by Facies")

for bar, recall in zip(bars, recalls):
    ax.text(
        recall + 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{recall:.2f}",
        va="center"
    )
plt.tight_layout()
plt.show()
