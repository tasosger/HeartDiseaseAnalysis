from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import math
from supplementaries import Vector,dot, squared_distance

def logistic(x: float)->float:
    return 1.0 / (1+math.exp(-x))

def logistic_prime(x: float)->float:
    y = logistic(x)
    return y*(1-y)


heart_disease = fetch_ucirepo(id=45)
data = heart_disease.data.features
target = heart_disease.data.targets
df = pd.DataFrame(data, columns=heart_disease.feature_names)
df.dropna(inplace=True)
df['target'] = target

df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])


for column in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    df[column] = (df[column] - df[column].mean()) / df[column].std()

df = df.sample(frac=1/1, random_state=0).reset_index(drop=True)
X = df.drop('target', axis=1).values
X = np.hstack([X, np.ones((X.shape[0], 1))])  
y = df['target'].values
y = np.eye(2)[y]  