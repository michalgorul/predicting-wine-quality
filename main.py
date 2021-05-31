import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier


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

    def correlationPlot(self):
        corr_matrix = self.data.corr()
        plt.figure(figsize=(11, 9))
        dropSelf = np.zeros_like(corr_matrix)
        dropSelf[np.triu_indices_from(dropSelf)] = True

        sns.heatmap(corr_matrix, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f",
                    mask=dropSelf)
        sns.set(font_scale=1.5)
        plt.show()

    def distributionOfVariables(self):
        plt.figure(figsize=(20, 22))

        for i in range(1, 12):
            plt.subplot(5, 4, i)
            sns.distplot(self.data[self.data.columns[i]], fit=norm)
        plt.show()

    def howManyQualityValues(self):
        print(self.data['quality'].value_counts().sort_values())

    def categoriseNumbers(self):
        self.data['quality'] = self.data['quality'].map(
            {3: 'low', 4: 'low', 5: 'medium', 6: 'medium', 7: 'high', 8: 'high'})
        self.data['quality'] = self.data['quality'].map({'low': 0, 'medium': 1, 'high': 2})

    def viewOutliers(self):
        sns.set()
        plt.figure(figsize=(30, 15))
        sns.boxplot(data=self.data)
        plt.show()

    def viewOutliersFromColumn(self, *columns):
        fig, ax = plt.subplots(1, 3)
        i = 0
        for column in columns:
            sns.boxplot(self.data[column], ax=ax[i])
            i += 1

        plt.show()

    def removeOutliers(self, * columns):
        for column in columns:
            lower = self.data[column].mean() - 3 * self.data[column].std()
            upper = self.data[column].mean() + 3 * self.data[column].std()
            self.data = self.data[(self.data[column] > lower) & (self.data[column] < upper)]

    def viewBestFeatures(self):
        x = self.data.drop("quality", axis=True)
        y = self.data["quality"]
        model = ExtraTreesClassifier()
        model.fit(x, y)
        print(model.feature_importances_)
        feat_importances = pd.Series(model.feature_importances_, index=x.columns)
        feat_importances.nlargest(9).plot(kind="barh")
        plt.show()


red = RedWine("C:\\Users\\Michael\\PycharmProjects\\BIAI\\winequality-red.csv")

# red.basicInfo()
#
# # fixed acidity impact on quality
# red.impactOnQuality("fixed acidity")
#
# # volatile acidity impact on quality
# red.impactOnQuality("volatile acidity")
#
# # cidric acid impact on quality
# red.impactOnQuality("citric acid")
#
# # residual sugar impact on quality
# red.impactOnQuality("residual sugar")
#
# # chlorides impact on quality
# red.impactOnQuality("chlorides")
#
# # free sulfur dioxide impact on quality
# red.impactOnQuality("free sulfur dioxide")
#
# # total sulfur dioxide impact on quality
# red.impactOnQuality("total sulfur dioxide")
#
# # density impact on quality
# red.impactOnQuality("density")
#
# # pH impact on quality
# red.impactOnQuality("pH")
#
# # sulphates impact on quality
# red.impactOnQuality("sulphates")
#
# # alcohol impact on quality
# red.impactOnQuality("alcohol")
#
# # basic info about quality of wines
# red.infoAboutQuality()

# # correlation plot
# red.correlationPlot()

# # Distribution of Variables
# red.distributionOfVariables()

# # Show how many values depending on quality
# red.howManyQualityValues()

# Categorise numbers for low, medium and high quality
red.categoriseNumbers()
red.howManyQualityValues()

# # Viewing outliers
red.viewOutliers()

# We need to remove outliers from 3 columns (residual sugar, free sulfur dioxide, total sulfur dioxide)
# to get more accurate results
red.viewOutliersFromColumn("residual sugar", "free sulfur dioxide", "total sulfur dioxide")

# Removing outliers
red.removeOutliers("residual sugar", "free sulfur dioxide", "total sulfur dioxide")
red.viewOutliersFromColumn("residual sugar", "free sulfur dioxide", "total sulfur dioxide")
red.correlationPlot()

# Viewing best Features for our Model
red.viewBestFeatures()


