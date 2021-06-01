import collections
import warnings
from collections import Counter

# data visualisation
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.stats import norm, boxcox
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class RedWine:
    def __init__(self, filePath):
        self.data = pd.read_csv(filePath)
        self.x = pd.DataFrame.empty
        self.y = pd.DataFrame.empty
        self.test_size = 0.2
        self.results = []

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

    def skewness(self, *columns):
        print()
        for column in columns:
            (mu, sigma) = norm.fit(self.data[column])
            print("Mean value of {}: {}, sigma {}: {}".format(column, mu, column, sigma))
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.distplot(self.data[column], fit=norm, color="blue")
            plt.title(column.capitalize() + " Distplot", color="darkred")
            plt.subplot(1, 2, 2)
            stats.probplot(self.data[column], plot=plt)
            plt.show()

    def fixSkewness(self, *columns):
        for column in columns:
            self.data[column], lam_fixed_acidity = boxcox(self.data[column])

    def viewOutliers(self):
        sns.set()
        plt.figure(figsize=(30, 15))
        sns.boxplot(data=self.data)
        plt.show()

    def countAndRemoveOutliers(self, columns):
        outlier_indices = []

        for c in columns:
            # 1st quartile
            Q1 = np.percentile(self.data[c], 25)
            # 3st quartile
            Q3 = np.percentile(self.data[c], 75)
            # IQR
            IQR = Q3 - Q1
            # Outlier Step
            outlier_step = IQR * 1.5
            # detect outlier and their indeces
            outlier_list_col = self.data[(self.data[c] < Q1 - outlier_step) | (self.data[c] > Q3 + outlier_step)].index
            # store indeces
            outlier_indices.extend(outlier_list_col)

        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1.5)

        print("number of outliers detected: ", len(multiple_outliers))
        # removal of outliers
        self.data = self.data.drop(multiple_outliers, axis=0).reset_index(drop=True)

    def viewOutliersFromColumn(self, *columns):
        fig, ax = plt.subplots(1, len(columns))
        i = 0
        for column in columns:
            sns.boxplot(self.data[column], ax=ax[i])
            i += 1

        plt.show()

    def howManyQualityValues(self):
        print(self.data['quality'].value_counts().sort_values())

    def categoriseNumbers(self):
        bins = (2, 6.5, 8)
        labels = [0, 1]
        self.data['quality'] = pd.cut(x=self.data['quality'], bins=bins, labels=labels)

    def smote(self):
        y = self.data.quality
        x = self.data.drop(["quality"], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size,
                                                                                random_state=200)
        sm = SMOTE(random_state=14)
        self.x_train_sm, self.y_train_sm = sm.fit_resample(self.x_train, self.y_train)
        print("Before smote: ", collections.Counter(self.y_train))
        print("After smote: ", collections.Counter(self.y_train_sm))

    def kNeighborsClassifier(self):
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(self.x_train_sm, self.y_train_sm)
        y_pred = knn.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)

        acc = accuracy_score(self.y_test, y_pred)
        score = knn.score(self.x_test, self.y_test)
        self.results.append(acc)

        print("Score : ", score)
        print("KNeighborsClassifier Acc : ", acc)

        plot_confusion_matrix(knn, self.x_test, self.y_test, cmap="Greens")
        plt.show()
        print(" \t \t  KNN Classification Report")
        print(classification_report(self.y_test, y_pred))

    def gradientBoostingClassifier(self):
        gbc = GradientBoostingClassifier(max_depth=6, random_state=2)
        gbc.fit(self.x_train_sm, self.y_train_sm)
        y_pred_gbc = gbc.predict(self.x_test)
        cm_aaa = confusion_matrix(self.y_test, y_pred_gbc)
        acc = accuracy_score(self.y_test, y_pred_gbc)
        score = gbc.score(self.x_test, self.y_test)
        self.results.append(acc)

        print("Score : ", score)
        print("GradientBoostingClassifier Acc : ", acc)

        plot_confusion_matrix(gbc, self.x_test, self.y_test, cmap="binary")
        plt.show()
        print(" \t \t  GradientBoostingClassifier Classification Report")
        print(classification_report(self.y_test, y_pred_gbc))

    def svc(self):
        svc = SVC()
        svc.fit(self.x_train_sm, self.y_train_sm)
        pred_svc = svc.predict(self.x_test)

        cm_svc = confusion_matrix(self.y_test, pred_svc)
        acc = accuracy_score(self.y_test, pred_svc)
        score = svc.score(self.x_test, self.y_test)
        self.results.append(acc)

        print("Score : ", score)
        print("SVC Acc : ", acc)

        plot_confusion_matrix(svc, self.x_test, self.y_test, cmap="Reds")
        plt.show()
        print(" \t \t  SVC Classification Report")
        print(classification_report(self.y_test, pred_svc))

    def xgb(self):
        xgb = XGBClassifier()
        xgb.fit(self.x_train_sm, self.y_train_sm)
        pred_xgb = xgb.predict(self.x_test)

        cm_aaa = confusion_matrix(self.y_test, pred_xgb)
        acc = accuracy_score(self.y_test, pred_xgb)
        score = xgb.score(self.x_test, self.y_test)
        self.results.append(acc)

        print("Score : ", score)
        print("XGBClassifier Acc : ", acc)

        plot_confusion_matrix(xgb, self.x_test, self.y_test, cmap="copper")
        plt.show()
        print(" \t \t  XGBClassifier Classification Report")
        print(classification_report(self.y_test, pred_xgb))

    def randomForestClassifier(self):
        rf = RandomForestClassifier(max_depth=18, random_state=44, bootstrap=False)
        rf.fit(self.x_train_sm, self.y_train_sm)
        y_pred_rf = rf.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred_rf)

        acc = accuracy_score(self.y_test, y_pred_rf)
        score = rf.score(self.x_test, self.y_test)
        self.results.append(acc)

        print("Score : ", score)
        print("RandomForestClassifier Acc : ", acc)

        plot_confusion_matrix(rf, self.x_test, self.y_test, cmap="pink")
        plt.show()

    def catBoostClassifier(self):
        parameters = {
            'depth': sp_randInt(4, 10),
            'learning_rate': sp_randFloat(),
            'iterations': sp_randInt(10, 100)
        }
        cat = CatBoostClassifier(iterations=1000, verbose=False, depth=8)
        randm = RandomizedSearchCV(estimator=cat, param_distributions=parameters,
                                   cv=2, n_iter=10, n_jobs=-1)
        randm.fit(self.x_train_sm, self.y_train_sm)

        pred_cat = randm.predict(self.x_test)

        cm_cat = confusion_matrix(self.y_test, pred_cat)
        acc = accuracy_score(self.y_test, pred_cat)
        score = randm.score(self.x_test, self.y_test)
        self.results.append(acc)

        print("Score : ", score)
        print("Basic KNN Acc : ", acc)

        plot_confusion_matrix(randm, self.x_test, self.y_test, cmap="hot")
        plt.show()
        print(" \t \t  CatBoostClassifier Classification Report")
        print(classification_report(self.y_test, pred_cat))

    def modelResult(self):
        df_result = pd.DataFrame({"Score": self.results, "ML Models": ["KNN", "GradientBoostingClassifier",
                                                                       "SVC", "XGBClassifier", "CatBoostClassifier",
                                                                       "RandomForestClassifier"]})
        print(df_result)
        g = sns.barplot("Score", "ML Models", data=df_result, palette='BrBG')
        g.set_xlabel("Score")
        g.set_title("Classifier Model Results", color="Black")
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

# # View skewness on graphs
# red.skewness("fixed acidity", "residual sugar", "free sulfur dioxide", "total sulfur dioxide", "alcohol")
#
# # Trying to eliminate skewness by using box cox
# red.fixSkewness("fixed acidity", "residual sugar", "free sulfur dioxide", "total sulfur dioxide", "alcohol")
#
# # View corrected skewness on graphs
# red.skewness("fixed acidity", "residual sugar", "free sulfur dioxide", "total sulfur dioxide", "alcohol")

# # Viewing outliers

# red.viewOutliers()
red.countAndRemoveOutliers(red.data.columns[:-1])
# red.viewOutliers()

# Show how many values depending on quality
red.howManyQualityValues()

# Categorise numbers for low and high quality
red.categoriseNumbers()
red.howManyQualityValues()
# We can see the difference between those two

# Lets balance our data
red.smote()
# some models
# KNeighborsClassifier
red.kNeighborsClassifier()

# GradientBoostingClassifier
red.gradientBoostingClassifier()

# SVC
red.svc()

# XGBClassifier
red.xgb()

# CatBoostClassifier
red.catBoostClassifier()

# RandomForestClassifier
red.randomForestClassifier()

red.modelResult()
