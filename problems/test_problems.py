"""
Defines test problems and initial iterates to be imported into other scripts to test optimisation algorithms.

Can also include solutions, if known.
"""

import autograd.numpy as np
from scipy.optimize import rosen

class Objective:
    def __init__(self, input_dim, func, x_sol=None, f_sol=None):
        self.input_dim = input_dim # Input dimension
        self.func = func # callable, returns objective value
        self.x_sol = x_sol
        self.f_sol = f_sol

# Using standard x0, see Appendix B of Dennis and Schnabel textbook.
# Note that this scipy version of an extended Rosenbrock function has
# multiple stationary points in high-dimensions! For details, see the paper
# https://dl.acm.org/doi/abs/10.1162/evco.2009.17.3.437
# The starting point seems to have a significant influence on whether the minimum is found or not.
def rosenbrock(input_dim):
    x0 = np.ones(input_dim, dtype='float32')
    x0[::2] = -1.2 # assign -1.2 to every other entry in x0

    x_sol = np.ones(input_dim)
    f_sol = 0
    return x0, Objective(input_dim, rosen, x_sol, f_sol)

# See https://www.sfu.ca/~ssurjano/powell.html
def powell(input_dim): # NOTE: input_dim must be multiple of 4
    if input_dim % 4 != 0:
        raise Exception('input_dim must be multiple of 4.')
    input_dim = int(input_dim)
    x0 = np.tile(np.array([3.0, -1.0, 0.0, 1.0]), input_dim // 4)
    def func(x):
        return np.sum([(x[4*i-3 -1] + 10 * x[4*i-2 -1])**2 + 5 * (x[4*i-1 -1] - x[4*i -1])**2 + (x[4*i-2 -1] - 2 * x[4*i-1 -1])**4 + 10 * (x[4*i-3 -1] - x[4*i -1])**4 for i in range(1, input_dim // 4 + 1)])
    x_sol = np.zeros(input_dim)
    f_sol = 0
    return x0, Objective(input_dim, func, x_sol, f_sol)

def well_conditioned_convex_quadratic(input_dim):
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        return 0.5 * sum(x ** 2)
    x_sol = np.zeros(input_dim)
    f_sol = 0
    return x0, Objective(input_dim, func, x_sol, f_sol)

def ill_conditioned_convex_quadratic(input_dim):
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        out = 0
        for i in range(input_dim):
            out += ((i + 1)**5) * x[i] ** 2
        return 0.5 * out
    x_sol = np.zeros(input_dim)
    f_sol = 0
    return x0, Objective(input_dim, func, x_sol, f_sol)

def select_problem(problem_name: str, input_dim: int):
    match problem_name:
        case 'rosenbrock':
            return rosenbrock(input_dim)
        case 'powell':
            return powell(input_dim)
        case 'well_conditioned_convex_quadratic':
            return well_conditioned_convex_quadratic(input_dim)
        case 'ill_conditioned_convex_quadratic':
            return ill_conditioned_convex_quadratic(input_dim)
            