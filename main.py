import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

winequality_red_csv = "C:\\Users\\Michael\\PycharmProjects\\BIAI\\winequality-red.csv"
data = pd.read_csv(winequality_red_csv)

# print(data.head())

# print(data.columns)

# print("Data Shape --> ", data.shape)

# print(data.describe())

# fixed acidity impact on quality
print(data[["fixed acidity", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="fixed acidity", data=data)
plt.title("Fixed Acidity impact on quality")
plt.grid()
plt.show()

# volatile acidity impact on quality
print(data[["volatile acidity", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="volatile acidity", data=data)
plt.title("Volatile Acidity impact on quality")
plt.grid()
plt.show()

# cidric acid impact on quality
print(data[["citric acid", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="citric acid", data=data)
plt.title("Citric Acid impact on quality")
plt.grid()
plt.show()

# residual sugar impact on quality
print(data[["residual sugar", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="residual sugar", data=data)
plt.title("Residual Sugar impact on quality")
plt.grid()
plt.show()

# chlorides impact on quality
print(data[["chlorides", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="chlorides", data=data)
plt.title("Chlorides impact on quality")
plt.grid()
plt.show()

# free sulfur dioxide impact on quality
print(data[["free sulfur dioxide", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="free sulfur dioxide", data=data)
plt.title("Free sulfur dioxide impact on quality")
plt.grid()
plt.show()

# total sulfur dioxide impact on quality
print(data[["total sulfur dioxide", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="total sulfur dioxide", data=data)
plt.title("Total sulfur dioxide impact on quality")
plt.grid()
plt.show()

# density impact on quality
print(data[["density", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="density", data=data)
plt.title("Density impact on quality")
plt.grid()
plt.show()

# pH impact on quality
print(data[["pH", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="pH", data=data)
plt.title("pH impact on quality")
plt.grid()
plt.show()

# sulphates impact on quality
print(data[["sulphates", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="sulphates", data=data)
plt.title("Sulphates against Quality")
plt.grid()
plt.show()

# alcohol impact on quality
print(data[["alcohol", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
sns.lineplot(x="quality", y="alcohol", data=data)
plt.title("Alcohol against Quality")
plt.grid()
plt.show()

# how many null values in columns
print(data.isnull().sum())
