import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/facies_data.csv")
print(df["Well Name"].unique())
# Выбираем скважину
well = "SHRIMPLIN"
# Оставляем только её данные
well_data = df[df["Well Name"] == well]
# Сортируем по глубине
well_data = well_data.sort_values("Depth")

print(well_data.head())
print(well_data.shape)

fig, axes = plt.subplots(1, 3,figsize=(18, 10),sharey=True)

# GR
axes[0].scatter(well_data["GR"],well_data["PHIND"],c=well_data["Facies"])

axes[0].set_xlabel("GR")
axes[0].set_ylabel("PHIND")
axes[0].set_title("GR")

# ILD
axes[1].scatter(well_data["ILD_log10"],well_data["PHIND"],c=well_data["Facies"])

axes[1].set_xlabel("ILD_log10")
axes[1].set_title("PHIND")

# DeltaPHI
axes[2].scatter(well_data["DeltaPHI"],well_data["PHIND"],c=well_data["Facies"])

axes[2].set_xlabel("DeltaPHI")
axes[2].set_title("DeltaPHI")


# Автоматически расставляем элементы
plt.tight_layout()
plt.show()