"""
In this module we implement quasi-Newton methods, in particular of the unlimited-memory sort (i.e., we do not strive to save up on storage of secant pairs, etc., as one may do with a method like L-BFGS).

Throughout, we denote, at iteration k, an approximation to the Hessian by B_k, and the inverse of B_k by H_k. Using Sherman-Morrison-Woodbury formula, we can directly update the inverse H_k as opposed to B_k, sidestepping the solution of a linear system in determining the search direction.
"""
from dataclasses import dataclass
from autograd import grad
import autograd.numpy as np

class BFGSLinesearchConfig:
    obj: any
    
    t_init: float = 1
    tol: float = 1e-6
    max_iter: int = 1_000
    iter_print_gap: int = 20
    verbose: bool = False

class BFGSLinesearch:
    def __init__(self, config: BFGSLinesearchConfig):
        for key, value in config.__dict__.items():
            setattr(self, key, value)
        
        self.func = self.obj.func
        self.grad_func = grad(self.func)
    
    def optimise(self, x0: np.ndarray, H0: np.ndarray):
        x = x0
        f_x = self.func(x)
        grad_vec = self.grad_func(x)
        grad_norm = np.linalg.norm(grad_vec)

        # For later plotting
        f_vals_list = [f_x]
        update_norms_list = []
        angles_to_grad_list = []
        grad_norms_list = [grad_norm]