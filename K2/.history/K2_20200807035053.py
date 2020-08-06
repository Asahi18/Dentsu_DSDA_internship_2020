import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
np.random.seed(2)

fname = '/Users/kitanoasahi/Documents/GitHub/Dentsu_DSDA_internship_2020/K2/Price.csv'
df = pd.read_csv(fname)