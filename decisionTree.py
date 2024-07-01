from typing import List
import math
from supplementaries import Vector
def entropy(class_probabilities: Vector)->float:
    return sum(-p*math.log(p,2) for p in class_probabilities if p>0)