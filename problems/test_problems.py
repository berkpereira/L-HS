"""
Defines test problems and initial iterates to be imported into other scripts to test optimisation algorithms.

Can also include solutions, if known.
"""

import autograd.numpy as np
import json
import pycutest
from autograd import grad, hessian
from scipy.optimize import rosen

class Objective:
    def __init__(self, name: str, input_dim: int,
                 func: callable, grad_func: callable=None, hess_func: callable=None,
                 x_sol=None, f_sol=None):
        # Basic data
        self.name = name
        self.input_dim = input_dim # aka "ambient dimension"
        
        # Problem callables
        self.func = func
        if grad_func is None:
            self.grad_func = grad(self.func)
        else:
            self.grad_func = grad_func
        if hess_func is None:
            self.hess_func = hessian(self.func)
        else:
            self.hess_func = hess_func
        
        # Known solution
        self.x_sol = x_sol
        self.f_sol = f_sol if f_sol is not None else self.get_best_known_f_sol()
    
    def get_best_known_f_sol(self):
        with open('results/best_known_results.json', 'r') as f:
            best_known_results = json.load(f)
        return best_known_results.get(self.name, None)

def import_cutest_problem(problem_name: str, input_dim: int=None):
    """
    Import a CUTEst problem by name using pycutest.

    :param problem_name: Name of the CUTEst problem.
    :return: Tuple (x0, Objective instance)
    """
    if input_dim is None:
        p = pycutest.import_problem(problem_name)
        input_dim = p.n
    else:
        p = pycutest.import_problem(problem_name, sifParams={'N': input_dim})
    func = p.obj
    grad_func = p.grad
    hess_func = p.hess

    x0 = p.x0

    f_sol = None
    
    full_name = f'{problem_name}_n{input_dim}'
    return x0, Objective(name=full_name, input_dim=input_dim, func=func,
                         grad_func=grad_func, hess_func=hess_func, f_sol=f_sol)

# Using standard x0 as reported in Appendix B of Dennis and Schnabel textbook.
# Note that this scipy version of an extended Rosenbrock function has
# multiple stationary points! For details, see 'Variant B' in the paper
# https://dl.acm.org/doi/abs/10.1162/evco.2009.17.3.437
# The starting point seems to have a significant influence on whether the
# minimum or some other stationary point is found.
def rosenbrock_multiple(input_dim: int):
    x0 = np.ones(input_dim, dtype='float32')
    x0[::2] = -1.2 # assign -1.2 to every other entry in x0

    x_sol = np.ones(input_dim)
    f_sol = 0
    return x0, Objective(f'rosenbrock_multiple_n{input_dim}', input_dim, rosen,
                         x_sol=x_sol, f_sol=f_sol)

# Using standard x0 as reported in Appendix B of Dennis and Schnabel textbook.
# This Rosebrock variant is only defined for even input space dimension.
# There is a single stationary point, the global minimiser, at x = ones.
# For details, see 'Variant A' in the paper
# https://dl.acm.org/doi/abs/10.1162/evco.2009.17.3.437.
def rosenbrock_single(input_dim: int):
    if input_dim % 2 != 0:
        raise Exception('This extended Rosenbrock variant is only defined for EVEN input space dimension!')
    x0 = np.ones(input_dim, dtype='float32')
    x0[::2] = -1.2
    
    x_sol = np.ones(input_dim)
    f_sol = 0

    def func(x: np.ndarray):
        return np.sum(100.0 * (x[::2]**2.0 - x[1::2])**2.0 + (x[::2] - 1)**2.0)
    return x0, Objective(f'rosenbrock_single_n{input_dim}', input_dim, func,
                         x_sol=x_sol, f_sol=f_sol)

# See https://www.sfu.ca/~ssurjano/powell.html
def powell(input_dim: int): # NOTE: input_dim must be multiple of 4
    if input_dim % 4 != 0:
        raise Exception('input_dim must be multiple of 4.')
    input_dim = int(input_dim)
    x0 = np.tile(np.array([3.0, -1.0, 0.0, 1.0]), input_dim // 4)
    def func(x):
        return np.sum([(x[4*i-3 -1] + 10 * x[4*i-2 -1])**2 + 5 * (x[4*i-1 -1] - x[4*i -1])**2 + (x[4*i-2 -1] - 2 * x[4*i-1 -1])**4 + 10 * (x[4*i-3 -1] - x[4*i -1])**4 for i in range(1, input_dim // 4 + 1)])
    x_sol = np.zeros(input_dim)
    f_sol = 0
    return x0, Objective(f'powell_n{input_dim}', input_dim, func,
                         x_sol=x_sol, f_sol=f_sol)

# "Perfect" convex quadratic. f(x) = squared_euclidian_norm(x)
def well_conditioned_convex_quadratic(input_dim: int):
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        return 0.5 * sum(x ** 2)
    x_sol = np.zeros(input_dim)
    f_sol = 0
    return x0, Objective(f'well_conditioned_convex_quadratic_n{input_dim}', input_dim, func,
                         x_sol=x_sol, f_sol=f_sol)

# Ill-conditioned convex quadratic where eigenvalues of the Hessian are
# logarithmically uniformly distributed from 1 to 10^4.
# Condition number of the Hessian is clearly kappa = 10^4.
def ill_conditioned_convex_quadratic(input_dim: int):
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        out = 0
        for i in range(input_dim):
            coefficient = 10 ** (4 * i / (input_dim - 1))
            out += coefficient * x[i] ** 2
        return 0.5 * out
    x_sol = np.zeros(input_dim)
    f_sol = 0
    return x0, Objective(f'ill_conditioned_convex_quadratic_n{input_dim}', input_dim, func,
                         x_sol=x_sol, f_sol=f_sol)

################################################################################
#################################  CUTEst  #####################################
################################################################################

# NOTE: NONDIA IS a sum-of-squares objective!
def nondia(input_dim: int):
    valid_dims = [10, 20, 30, 90, 100, 500, 1000, 5000, 10000]
    if input_dim not in valid_dims:
        raise Exception(f'input/ambient dimension must be from valid list!\nThat is {valid_dims}.')

    p = pycutest.import_problem('NONDIA', sifParams={'N': input_dim})
    func = p.obj
    grad_func = p.grad
    hess_func = p.hess
    
    x0 = p.x0
    
    f_sol = None # not sure about other input dimensions
    
    return x0, Objective(f'nondia_n{input_dim}', input_dim, func, grad_func, hess_func,
                         f_sol=f_sol)

# NOTE: GENHUMPS is NOT a sum-of-squares objective!
def genhumps(input_dim: int):
    valid_dims = [5, 10, 100, 500, 1000, 5000]
    if input_dim not in valid_dims:
        raise Exception(f'input/ambient dimension must be from valid list!\nThat is {valid_dims}.')

    p = pycutest.import_problem('GENHUMPS', sifParams={'N': input_dim})
    func = p.obj
    grad_func = p.grad
    hess_func = p.hess
    
    x0 = p.x0
    
    # Provisional f_sol
    f_sol = None
    
    return x0, Objective(f'genhumps_n{input_dim}', input_dim, func, grad_func,
                         hess_func, f_sol=f_sol)


################################################################################
################################################################################
################################################################################

def select_problem(problem_name: str, input_dim: int=None):
    # Load CUTEst problem names from JSON file
    with open('problems/cutest_unconstrained.json', 'r') as f:
        cutest_unconstrained_names = json.load(f)

    if problem_name in cutest_unconstrained_names:
        return import_cutest_problem(problem_name, input_dim)

    # Non-CUTEst problems:
    match problem_name:
        case 'rosenbrock_single':
            return rosenbrock_single(input_dim)
        case 'rosenbrock_multiple':
            return rosenbrock_multiple(input_dim)
        case 'powell':
            return powell(input_dim)
        case 'well_conditioned_convex_quadratic':
            return well_conditioned_convex_quadratic(input_dim)
        case 'ill_conditioned_convex_quadratic':
            return ill_conditioned_convex_quadratic(input_dim)
        case _:
            raise ValueError(f"Problem {problem_name} is not defined.")
            