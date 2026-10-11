import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("../facies_data.csv")
print(df["Well Name"].unique())

well = "SHRIMPLIN"
# Оставляем только её данные
well_data = df[df["Well Name"] == well]

# Сортируем по глубине
well_data = well_data.sort_values("Depth")

print(well_data.head())
print(well_data.shape)

#GR по глубине
plt.plot(well_data["GR"], well_data["Depth"])

plt.xlabel("GR")
plt.ylabel("Depth")

# Глубина увеличивается вниз
plt.gca().invert_yaxis()
plt.show()
fig, axes = plt.subplots(1, 6,figsize=(18, 10),sharey=True)

# GR
axes[0].plot(well_data["GR"],well_data["Depth"])

axes[0].set_xlabel("GR")
axes[0].set_ylabel("Depth")
axes[0].set_title("GR")

# ILD
axes[1].plot(well_data["ILD_log10"],well_data["Depth"])

axes[1].set_xlabel("ILD_log10")
axes[1].set_title("ILD")

#DeltaPHI
axes[2].plot(well_data["DeltaPHI"],well_data["Depth"])

axes[2].set_xlabel("DeltaPHI")
axes[2].set_title("DeltaPHI")



# PHIND
axes[3].plot(well_data["PHIND"],well_data["Depth"])

axes[3].set_xlabel("PHIND")
axes[3].set_title("PHIND")

#PE
axes[4].plot(well_data["PE"],well_data["Depth"])

axes[4].set_xlabel("PE")
axes[4].set_title("PE")

#Facies

axes[5].scatter(well_data["Facies"],well_data["Depth"],c=well_data["Facies"])

axes[5].set_xlabel("Facies")
axes[5].set_title("Facies")


# Глубина увеличивается вниз
axes[0].invert_yaxis()

# Заголовок всего рисунка
fig.suptitle(f"Каротажные кривые — скважина {well}",fontsize=16)

# Автоматически расставляем элементы
plt.tight_layout()
plt.show()