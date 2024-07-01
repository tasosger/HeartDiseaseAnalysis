from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import math
from typing import List
from supplementaries import Vector,dot, squared_distance, vector_sum

def logistic(x: float)->float:
    return 1.0 / (1+math.exp(-x))

def logistic_prime(x: float)->float:
    y = logistic(x)
    return y*(1-y)
def _negative_log_likelihood(x: Vector, y: float, beta: Vector)-> float:
    if y==1:
         return -math.log(logistic(dot(x,beta)))
    else:
        return -math.log(1-logistic(dot(x,beta)))
def negative_log_likelihood(xs: List[Vector], ys: List[Vector], beta: Vector)-> float:
    return sum(_negative_log_likelihood(x,y,beta) for x,y in zip(xs,ys))

def _negative_log_partial_j(x: Vector, y: float, beta: Vector, j : int)-> float:
    return -(y-logistic(dot(x,beta)))*x[j]
def _negative_log_gradient(x: Vector, y: float, beta: Vector)->Vector:
    return [_negative_log_partial_j((x,y,beta,j) for j in range(len(beta)))]
def negative_log_gradient(xs: List[Vector], ys: List[ Vector], beta: Vector)-> Vector:
    return vector_sum([_negative_log_gradient(x,y,beta) for x,y in zip(xs,ys)])

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
 
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]