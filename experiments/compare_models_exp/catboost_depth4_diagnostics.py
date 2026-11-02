
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "facies_data.csv"
RESULTS_DIR = ROOT / "results" / "logo_6"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
features = [
    "GR", "ILD_log10", "DeltaPHI",
    "PHIND", "PE", "RELPOS", "NM_M"
]
classes = list(range(1, 10))
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
dev_idx, holdout_idx = next(
    splitter.split(df, df["Facies"], df["Well Name"])
)
dev = df.iloc[dev_idx].copy()

X = dev[features]
y = dev["Facies"]
groups = dev["Well Name"]

cv = LeaveOneGroupOut()
oof_pred = np.full(len(dev), -1, dtype=int)
oof_conf = np.full(len(dev), np.nan)

for train_idx, val_idx in cv.split(X, y, groups):

    model = CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=2
    )
    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )
    pred = np.asarray(
        model.predict(X.iloc[val_idx])
    ).ravel().astype(int)
    probabilities = model.predict_proba(
        X.iloc[val_idx]
    )
    oof_pred[val_idx] = pred
    oof_conf[val_idx] = probabilities.max(axis=1)
assert np.all(oof_pred != -1)
print("OOF Accuracy:", round(
    accuracy_score(y, oof_pred), 4
))
print("OOF Macro F1:", round(
    f1_score(
        y, oof_pred,
        labels=classes,
        average="macro",
        zero_division=0
    ), 4
))

results = dev[
    ["Well Name", "Depth", "GR", "Facies"]
].copy()
results["Prediction"] = oof_pred
results["Confidence"] = oof_conf
results["Error"] = (
    results["Facies"] != results["Prediction"]
).astype(int)
results.sort_values(
    ["Well Name", "Depth"]
).to_csv(
    RESULTS_DIR / "CB_depth4_oof_predictions.csv",
    index=False
)

facies_cmap = ListedColormap(
    plt.colormaps["Greens"](
        np.linspace(0.25, 0.95, 9)
    )
)
facies_norm = BoundaryNorm(
    np.arange(0.5, 10.5, 1),
    facies_cmap.N
)
error_cmap = ListedColormap([
    "#e8f5e9",  # Правильный прогноз
    "#14532d"   # Ошибка
])
error_norm = BoundaryNorm(
    [-0.5, 0.5, 1.5],
    error_cmap.N
)

def plot_well(well_name):

    well = results[
        results["Well Name"] == well_name
    ].sort_values("Depth")

    depth = well["Depth"].to_numpy()

    # границы интервалов между точками измерения
    edges = np.empty(len(depth) + 1)

    edges[1:-1] = (
        depth[:-1] + depth[1:]
    ) / 2

    edges[0] = depth[0] - (
        depth[1] - depth[0]
    ) / 2

    edges[-1] = depth[-1] + (
        depth[-1] - depth[-2]
    ) / 2

    fig, axes = plt.subplots(
        1, 5,
        figsize=(14, 10),
        sharey=True,
        gridspec_kw={
            "width_ratios": [2.5, 1, 1, 0.8, 1.5]
        },
        constrained_layout=True
    )

    #дорожка 1: GR
    axes[0].plot(
        well["GR"], depth,
        color="#166534",
        linewidth=1.2
    )

    axes[0].set_title("GR")
    axes[0].set_xlabel("GR")
    axes[0].set_ylabel("Depth")
    axes[0].grid(alpha=0.2)
    axes[0].invert_yaxis()

    # дорожки 2 и 3: фации
    for ax, column, title in zip(
        axes[1:3],
        ["Facies", "Prediction"],
        ["True Facies", "Predicted Facies"]
    ):

        image = ax.pcolormesh(
            [0, 1],
            edges,
            well[column].to_numpy().reshape(-1, 1),
            cmap=facies_cmap,
            norm=facies_norm,
            shading="flat"
        )

        ax.set_title(title)
        ax.set_xticks([])

    # дорожка 4: ошибки
    axes[3].pcolormesh(
        [0, 1],
        edges,
        well["Error"].to_numpy().reshape(-1, 1),
        cmap=error_cmap,
        norm=error_norm,
        shading="flat"
    )

    axes[3].set_title("Errors")
    axes[3].set_xticks([])

    #дорожка 5: макс вероятность класса
    axes[4].plot(
        well["Confidence"],
        depth,
        color="#15803d",
        linewidth=1.2
    )

    axes[4].set_title("Max probability")
    axes[4].set_xlim(0, 1)
    axes[4].grid(alpha=0.2)

    fig.colorbar(
        image,
        ax=axes[1:3],
        ticks=classes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.04
    )
    well_accuracy = accuracy_score(
        well["Facies"],
        well["Prediction"]
    )
    fig.suptitle(
        f"{well_name} | OOF Accuracy={well_accuracy:.3f}"
    )

    output_path = RESULTS_DIR / f"CB_depth4_{well_name}.png"
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    print("Сохранён график:", output_path)
for well_name in ["SHANKLE", "NEWBY"]:
    plot_well(well_name)
print("\nГотово!")
for well_name in ["SHANKLE", "NEWBY"]:
    well = results[
        results["Well Name"] == well_name
    ]
    print("СКВАЖИНА:", well_name)
    # Матрица ошибок в виде таблицы
    matrix = pd.crosstab(
        well["Facies"],
        well["Prediction"]
    )
    #все девять классов
    matrix = matrix.reindex(
        index=classes,
        columns=classes,
        fill_value=0
    )
    print("\nСтроки — истинные фации")
    print("Столбцы — предсказанные фации")
    print(matrix.to_string())
    # отдельно песчаники
    sandstone = well[
        well["Facies"] == 1
    ]
    print("\nКоличество интервалов Facies 1:",
          len(sandstone))
    print("Во что модель классифицировала Facies 1:")
    print(
        sandstone["Prediction"]
        .value_counts()
        .sort_index()
        .to_string()
    )