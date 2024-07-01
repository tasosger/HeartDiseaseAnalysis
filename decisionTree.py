from typing import List, Any
import math
from supplementaries import Vector
from collections import Counter

def entropy(class_probabilities: Vector)->float:
    return sum(-p*math.log(p,2) for p in class_probabilities if p>0)

def class_probabilities(labels: List[Any])->List[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

