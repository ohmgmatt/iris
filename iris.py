import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split, KFold, \
                                    cross_val_score, StratifiedKFold





## Import the dataset
dataset = pd.read_csv('iris_working.csv', header =None,
                names = ['sepal_length', 'sepal_width', 'petal_length','petal_width','class'])
# print(dataset.head())
# print(dataset.describe())
features = dataset.values[:,0:4]
labels = dataset.values[:,4]
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=7)
