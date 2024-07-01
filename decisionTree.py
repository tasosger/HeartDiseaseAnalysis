from typing import List, Any, Dict, TypeVar, NamedTuple, Union
import math
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def entropy(class_probabilities: List[float]) -> float:
    return sum(-p * math.log(p, 2) for p in class_probabilities if p > 0)

def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

def partition_entropy(subsets: List[List[Any]]) -> float:
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

T = TypeVar('T')

def partition_by(inputs: List[Dict[str, Any]], attribute: str) -> Dict[Any, List[Dict[str, Any]]]:
    partitions: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for input in inputs:
        key = input[attribute]
        partitions[key].append(input)
    return partitions

def partition_entropy_by(inputs: List[Dict[str, Any]], attribute: str, label_attribute: str) -> float:
    partitions = partition_by(inputs, attribute)
    labels = [[input[label_attribute] for input in partition] for partition in partitions.values()]
    return partition_entropy(labels)

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]

def classify(tree: DecisionTree, input: Dict[str, Any]) -> Any:
    if isinstance(tree, Leaf):
        return tree.value

    subtree_key = input.get(tree.attribute)

    if subtree_key not in tree.subtrees:
        return tree.default_value

    subtree = tree.subtrees[subtree_key]
    return classify(subtree, input)

def build_tree_id3(inputs: List[Dict[str, Any]], split_attributes: List[str], target_attribute: str, max_depth: int) -> DecisionTree:
    label_counts = Counter(input[target_attribute] for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    if len(label_counts) == 1:
        return Leaf(most_common_label)
    if not split_attributes or max_depth == 0:
        return Leaf(most_common_label)

    def split_entropy(attribute: str) -> float:
        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    partitions = partition_by(inputs, best_attribute)
    subtrees = {attribute_value: build_tree_id3(subset, new_attributes, target_attribute, max_depth - 1)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)

def print_tree(tree: DecisionTree, indent: str = "") -> None:
    if isinstance(tree, Leaf):
        print(indent + "Leaf:", tree.value)
    else:
        print(indent + "Split:", tree.attribute)
        for key, subtree in tree.subtrees.items():
            print(indent + f" {key} ->")
            print_tree(subtree, indent + "  ")


heart_disease = fetch_ucirepo(id=45)
data = heart_disease.data.features
target = heart_disease.data.targets
df = pd.DataFrame(data, columns=heart_disease.feature_names)
df.dropna(inplace=True)
df['target'] = target
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df[numerical_features] = binner.fit_transform(df[numerical_features])


df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'] + numerical_features)


print("Class distribution in target variable:")
print(df['target'].value_counts())


train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


def cross_val_accuracy(data, target_column, split_attributes, max_depth, cv=5):
    data_records = data.to_dict(orient='records')
    folds = np.array_split(data, cv)
    accuracies = []

    for i in range(cv):
        validation_data = folds[i]
        training_data = pd.concat(folds[:i] + folds[i+1:])
        
        tree = build_tree_id3(training_data.to_dict(orient='records'), split_attributes, target_column, max_depth)
        test_records = validation_data.to_dict(orient='records')
        test_predictions = [classify(tree, record) for record in test_records]
        test_actual = validation_data[target_column].tolist()
        accuracy = sum(1 for pred, actual in zip(test_predictions, test_actual) if pred == actual) / len(test_actual)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)


split_attributes = list(df.columns)
split_attributes.remove('target')
max_depth = 5  


mean_accuracy, std_accuracy = cross_val_accuracy(df, 'target', split_attributes, max_depth)
print(f"Cross-Validation Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")

tree = build_tree_id3(train_data.to_dict(orient='records'), split_attributes, 'target', max_depth)


print("Decision tree structure:")
print_tree(tree)


test_records = test_data.to_dict(orient='records')
test_predictions = [classify(tree, record) for record in test_records]
test_actual = test_data['target'].tolist()


accuracy = sum(1 for pred, actual in zip(test_predictions, test_actual) if pred == actual) / len(test_actual)
print("Accuracy on testing set:", accuracy)