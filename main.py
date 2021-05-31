import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualisation
import matplotlib.pyplot as plt
import seaborn as sns


class RedWine:
    def __init__(self, filePath):
        self.data = pd.read_csv(filePath)

    def basicInfo(self):
        print("Basic information about data:\n")
        print(self.data.describe())
        print("Data Shape --> ", self.data.shape, "\n")
        print("How many null values in columns:")
        print(self.data.isnull().sum())

    def impactOnQuality(self, feature):
        print(self.data[[feature, "quality"]].groupby(["quality"], as_index=False).mean().sort_values(by="quality"))
        sns.lineplot(x="quality", y=feature, data=self.data)
        plt.title(feature.capitalize() + " impact on quality")
        plt.grid()
        plt.show()

    def infoAboutQuality(self):
        Number = self.data.quality.value_counts().values
        Label = self.data.quality.value_counts().index
        circle = plt.Circle((0, 0), 0.2, color="white")
        explodeTuple = (0.0, 0.0, 0.0, 0.3, 0.5, 0.5)

        plt.figure(figsize=(13, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(self.data["quality"])
        plt.xlabel("quality")
        plt.title("quality distribution", color="black", fontweight='bold', fontsize=11)
        plt.subplot(1, 2, 2)
        plt.pie(Number, labels=Label, autopct='%1.2f%%', explode=explodeTuple, startangle=60)
        p = plt.gcf()
        p.gca().add_artist(circle)
        plt.title("quality distribution", color="black", fontweight='bold', fontsize=11)
        plt.legend()
        plt.show()


red = RedWine("C:\\Users\\Michael\\PycharmProjects\\BIAI\\winequality-red.csv")

red.basicInfo()

# fixed acidity impact on quality
red.impactOnQuality("fixed acidity")

# volatile acidity impact on quality
red.impactOnQuality("volatile acidity")

# cidric acid impact on quality
red.impactOnQuality("citric acid")

# residual sugar impact on quality
red.impactOnQuality("residual sugar")

# chlorides impact on quality
red.impactOnQuality("chlorides")

# free sulfur dioxide impact on quality
red.impactOnQuality("free sulfur dioxide")

# total sulfur dioxide impact on quality
red.impactOnQuality("total sulfur dioxide")

# density impact on quality
red.impactOnQuality("density")

# pH impact on quality
red.impactOnQuality("pH")

# sulphates impact on quality
red.impactOnQuality("sulphates")

# alcohol impact on quality
red.impactOnQuality("alcohol")

# basic info about quality of wines
red.infoAboutQuality()
