"""
Defines test problems and initial iterates to be imported into other scripts to test optimisation algorithms.
"""

import numpy as np
from scipy.optimize import rosen

class Objective:
    def __init__(self, input_dim, func):
        self.input_dim = input_dim # Input dimension
        self.func = func # callable, returns objective value

def rosenbrock(input_dim):
    x0 = np.zeros(input_dim, dtype='float32')
    return x0, Objective(input_dim, rosen)

def powell_singular():
    input_dim = 4
    x0 = np.array([3.0, -1.0, 0.0, 1.0])
    def func(x):
        return (x[0] + 10*x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10 * (x[0] - x[3])**4
    return x0, Objective(input_dim, func)

def well_conditioned_convex_quadratic(input_dim):
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        out = 0
        for i in range(input_dim):
            out += (i + 1) * x[i] ** 2
        return 0.5 * out
    return x0, Objective(input_dim, func)

def ill_conditioned_convex_quadratic(input_dim):
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        out = 0
        for i in range(input_dim):
            out += ((i + 1)**5) * x[i] ** 2
        return 0.5 * out
    return x0, Objective(input_dim, func)

def select_problem(problem_name: str, input_dim: int):
    match problem_name:
        case 'rosenbrock':
            return rosenbrock(input_dim)
        case 'powell_singular':
            return powell_singular()
        case 'well_conditioned_convex_quadratic':
            return well_conditioned_convex_quadratic(input_dim)
        case 'ill_conditioned_convex_quadratic':
            return ill_conditioned_convex_quadratic(input_dim)
            