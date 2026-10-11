import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../facies_data.csv")

logs = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]

fig, axes = plt.subplots(1, 5, figsize=(20, 6))

for ax, log in zip(axes, logs):
    df.boxplot(
        column=log,
        by="Facies",
        ax=ax)

    ax.set_title(log)
    ax.set_xlabel("Facies")
    ax.set_ylabel(log)

fig.suptitle("")
plt.tight_layout()
plt.show()

logs = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]

summary = df.groupby("Facies")[logs].median()

print(summary)