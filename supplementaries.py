import math
from typing import List

Vector = List[float]

def subtract(v: Vector, w:Vector) -> Vector:
    assert len(v)==len(w)      
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def add(v: Vector, w:Vector) -> Vector:
    assert len(v)==len(w)

    return [v_i + w_i for v_i, w_i in zip(v,w)]
def scalar_multiply (c:float, v:Vector) -> Vector:
    return [c*v_i for v_i in v]

def dot(v:Vector, w:Vector) -> float:
    assert len(v)==len(w)
    return sum(v_i * w_i for v_i,w_i in zip(v,w))

def sum_of_squares(v:Vector)->float:
    return dot(v,v)

def squared_distance(v:Vector,w:Vector)->float:
    return sum_of_squares(subtract(v,w))


def gradient_step(v: Vector, gradient: Vector, step_size: float) ->float:
    assert len(gradient)==len(v)
    step = scalar_multiply(step_size,gradient)
    return add(v,step)

def vector_sum (vectors: List[Vector])-> Vector:
    dimensions = len(vectors[0])
    assert all(len(v)==dimensions for v in vectors)
    return [sum(vector[i] for vector in vectors) for i in range(dimensions)]