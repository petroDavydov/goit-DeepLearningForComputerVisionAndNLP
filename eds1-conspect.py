import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import warnings
# filter warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Module_1_Lecture_2_Class_Spaceship_Titanic.csv')
df = df.set_index('PassengerId')


print(df.info())
print(df.head(10))