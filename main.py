import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("C:\\Users\\Michael\\PycharmProjects\\BIAI\\winequality-red.csv")

# print(data.head())

# print(data.columns)

# print("Data Shape --> ", data.shape)

# print(data.describe())

# fixed acidity impact on quality
print(data[["fixed acidity", "quality"]].groupby(["quality"], as_index=False).mean().sort_values(
    by="quality"))

# volatile acidity impact on quality
print(data[["volatile acidity","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# cidric acid impact on quality
print(data[["citric acid","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# residual sugar impact on quality
print(data[["residual sugar","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# chlorides impact on quality
print(data[["chlorides","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# free sulfur dioxide impact on quality
print(data[["free sulfur dioxide","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# total sulfur dioxide impact on quality
print(data[["total sulfur dioxide","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# density impact on quality
print(data[["density","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# pH impact on quality
print(data[["pH","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# sulphates impact on quality
print(data[["sulphates","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))

# alcohol impact on quality
print(data[["alcohol","quality"]].groupby(["quality"], as_index = False).mean().sort_values(by = "quality"))