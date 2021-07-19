import numpy as np
from scipy.stats.stats import mode
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


titanic = sns.load_dataset('titanic')
titanic = titanic[['survived', 'pclass', 'sex', 'age']]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['male', 'female'], [0,1], inplace=True)
print(titanic.head())   

model = KNeighborsClassifier(n_neighbors=3)
y = titanic['survived']
X = titanic.drop('survived', axis=1)

model.fit(X, y)
print(model.score(X, y))

def survive(model, pclass=1, sex=1, age=2): #enter you data here (male : 0, female :1) 
    x =np.array([pclass, sex, age]).reshape(1, 3)
    print(model.predict(x))
    print(model.predict_proba(x))

survive(model)
