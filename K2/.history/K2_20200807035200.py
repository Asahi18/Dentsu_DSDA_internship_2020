import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
np.random.seed(2)

fname = './Price.csv'
df = pd.read_csv(fname)

price = df[['id','price']]
df = df.drop('price',axis=1)

# pre-processing
X = df.drop('date', axis=1)
X = X.drop('id', axis=1)

Y = price.drop('id', axis=1)

m = df.shape[0]
m1 = int(m * 0.7)
m2 = int(m * 0.85)

X_train = X.iloc[0:m1]
X_cv = X.iloc[m1:m2]
X_test = X.iloc[m2:]

Y_train = Y.iloc[0:m1]
Y_cv = Y.iloc[m1:m2]
Y_test = Y.iloc[m2:]

