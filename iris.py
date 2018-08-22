import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Import the dataset
data = pd.read_csv('iris_working.csv', header =None,
                names = ['sepal_length', 'sepal_width', 'petal_length','petal_width','class'])
# print(data.head())
# print(data.describe())
