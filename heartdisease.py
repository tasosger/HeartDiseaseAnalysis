from supplementaries import Vector,dot, squared_distance
import math
from typing import List
import random
import tqdm
from supplementaries  import gradient_step
random.seed(0)
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

def step_function(x: float)-> float:
    return 1.0 if x>=0 else 0.0



def sigmoid(t: float)-> float:
    return 1/ (1+math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    z = dot(weights, inputs)
    return sigmoid(z)
def feed_forward(neural_network: List[List[Vector]], input_vector: Vector)-> List[Vector]:
    outputs = []

    for i,layer in enumerate(neural_network):
        inputs_with_bias = input_vector +[1]
        
        output = [neuron_output(neuron, inputs_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs

def sqerror_gradients(network: List[List[Vector]], input_vector: Vector, target_vector: Vector)->List[List[Vector]]:
    hidden_outputs, outputs = feed_forward(network,input_vector)
    output_deltas = [output*(1-output)*(output-target) for output, target in zip(outputs,target_vector)]
    output_grads = [[output_deltas[i]*hidden_output for hidden_output in hidden_outputs + [1] ] for i, output_neuron in enumerate(network[-1])]
    hidden_deltas = [hidden_output * (1-hidden_output)*dot(output_deltas,[n[i] for n in network[-1]]) for i, hidden_output in enumerate(hidden_outputs)]
    hidden_grads = [[hidden_deltas[i]*input  for input in input_vector + [1]] for i, hidden_neuron in enumerate(network[0])]
    return [hidden_grads,output_grads]


def argmax(xs: list) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


    



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

NUM_HID = 25
network = [
    [[random.uniform(-1,1) for _ in range(X.shape[1])] for _ in range(NUM_HID)],  
    [[random.uniform(-1,1) for _ in range(NUM_HID + 1)] for _ in range(y.shape[1])]  
]



learning_rate = 1.0


with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y_true in zip(X, y):
            predicted = feed_forward(network, x)[-1]
            epoch_loss += squared_distance(predicted, y_true)
            gradients = sqerror_gradients(network, x, y_true)
            network = [
                [gradient_step(neuron, grad, -learning_rate) for neuron, grad in zip(layer, layer_grad)]
                for layer, layer_grad in zip(network, gradients)
            ]

num_correct = 0
for x, target in zip(X, y):
    val = feed_forward(network, x)[-1]
    print(val)
    predicted = argmax(val)
    actual = argmax(target)
    if predicted == actual:
        num_correct += 1
    print(actual,predicted)
accuracy = num_correct / len(X)
print(f"Accuracy: {accuracy * 100:.2f}%")