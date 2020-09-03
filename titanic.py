# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# %%
# get data and label 
train_dataFrame = pd.read_csv("data/train.csv")
test_dataFrame = pd.read_csv("data/test.csv")

# preview the data
train_dataFrame.head()
train_dataFrame.info()
test_dataFrame.info()


# %%
# throw the unnecessary data column

train_dataFrame = train_dataFrame.drop(['PassengerId','Name','Ticket'], axis=1)
test_dataFrame  = test_dataFrame.drop(['Name','Ticket'], axis=1)


# %%
# fill the missing value of the 'Embarked' column

train_dataFrame["Embarked"] = train_dataFrame["Embarked"].fillna("S")