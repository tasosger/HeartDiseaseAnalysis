from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import math
from typing import List
from supplementaries import Vector,dot, squared_distance, vector_sum,gradient_step, de_mean, vector_means
import random
import tqdm
from sklearn.model_selection import train_test_split
def predict(x: Vector, b: Vector) -> float:
    return dot(x, b)

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y 

def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]

def least_squares_fit(xs: List[Vector], ys: Vector, learning_rate: float = 0.001, num_steps: int = 1000, batch_size: int = 1) -> Vector:
    guess = [random.random() for _ in xs[0]]
    for _ in tqdm.trange(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]
            gradient = vector_means([sqerror_gradient(x, y, guess) for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)
    return guess

def total_sum_of_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))

def multiple_r_squared(xs: List[Vector], ys: List[float], beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('With Normalization: ')
beta = least_squares_fit(X_train, y_train)
print("Learned weights (beta):", beta)

r_squared_train = multiple_r_squared(X_train, y_train, beta)
print("R-squared on training set:", r_squared_train)

r_squared_test = multiple_r_squared(X_test, y_test, beta)
print("R-squared on testing set:", r_squared_test)


y_pred_test = [1 if predict(x, beta) > 0.5 else 0 for x in X_test]
accuracy = np.mean([pred == actual for pred, actual in zip(y_pred_test, y_test)])
print("Accuracy on testing set:", accuracy)


print('second approach:')
