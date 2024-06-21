"""
In this module we implement a (unlimited-memory) BFGS linesearch method.

Throughout, we denote, at iteration k, an approximation to the Hessian by B_k,
and the inverse of B_k by H_k. Using Sherman-Morrison-Woodbury formula, we can
directly update the inverse H_k as opposed to B_k, sidestepping the solution of
a linear system in determining the search direction.

The BFGS implementation follows the description from the well-known textbook
Numerical Optimization by Nocedal and Wright, 2nd ed., Chapter 6.1.

TODO:
Improve linesearch algorithm --- the scipy implementation seems to fail on a
(very) badly conditioned quadratic. May want to consider the Hager-Zhang
linesearch algorithm, which seems to be the state of the art (See reference
paper: https://epubs.siam.org/doi/abs/10.1137/030601880. Also see
implementation within TensorFlow:
https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/optimizer/linesearch/hager_zhang.py)

When we refer to a heuristic of rescaling H0, this refers to a technique used
in practice when the initial H0 provided is simply an identity matrix.
This is presented in Nocedal & Wright, 2nd ed., Sec. 6.1 (Implementation).
"""

from dataclasses import dataclass
from autograd import grad
from solvers.utils import SolverOutput
import autograd.numpy as np
import scipy.optimize
from .utils import strong_wolfe_linesearch

@dataclass
class BFGSLinesearchConfig:
    obj: any
    H0_rescale_heuristic: bool = True # Whethere to use rescaling heuristic of H0.
    c1: float = 1e-4 # Armijo condition scaling
    c2: float = 0.9  # Strong curvature condition scaling
    linesearch_max_iter: int = 10 # Number of max iterations in the linesearch
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

    # We use scipy's implementation of a strong-Wolfe-condition-ensuring
    # linesearch (provided direction is a descent direction, of course)
    def strong_wolfe_linesearch(self, x, direction):
        return strong_wolfe_linesearch(func=self.func, grad_func=self.grad_func,
                                       x=x, direction=direction, c1=self.c1,
                                       c2=self.c2, max_iter=self.linesearch_max_iter)
    
    def optimise(self, x0: np.ndarray, H0: np.ndarray):
        # Initialise with initial inputs given
        x = x0
        H = H0

        # Check whether starting H0 is the identity.
        if np.array_equal(H0, np.identity(self.obj.input_dim)):
            H0_is_eye = True
        else:
            H0_is_eye = False

        # Check that initial H is symmetric PD
        if (not np.allclose(H, np.transpose(H))) or np.any(np.linalg.eig(H)[0] <= 0):
            raise Exception('Initial H matrix must be symmetric positive definite!')
        
        x_next = x
        H_next = H
        f_next = self.func(x_next)
        grad_next = self.grad_func(x_next)
        
        # For later plotting
        f_vals_list = []
        update_norms_list = []
        angles_to_grad_list = []
        grad_norms_list = []

        for k in range(self.max_iter):
            # Assign updated quantities
            x = x_next
            H = H_next
            f_x = f_next
            grad_vec = grad_next
            grad_norm = np.linalg.norm(grad_vec)

            # Compute search direction
            direction = - H @ grad_vec
            # Compute step size
            # TODO (not vital): use some of the output arguments of this linesearch
            # function, which returns quantities we do actually need (currently
            # some of this is being computed twice at each iteration)
            step_size, _, _, _, _, _ = self.strong_wolfe_linesearch(x=x,direction=direction)

            if step_size is None:
                raise Exception('Strong Wolfe conditions linesearch failed within the alloted max iterations!')

            if self.verbose and k % self.iter_print_gap == 0:
                x_str = ", ".join([f"{xi:7.4f}" for xi in x])
                print(f"k = {k:4} || x = [{x_str}] || f(x) = {f_x:6.6e} || g_norm = {grad_norm:6.6e} || t = {step_size:8.6f}")

            # Set next iterate; do NOT reassign (yet)
            x_next = x + direction * step_size
            f_next = self.func(x_next)
            grad_next = self.grad_func(x_next)
            
            # Compute secant pair
            x_diff = x_next - x
            grad_diff = grad_next - grad_vec

            # Use heuristic to rescale H0 after the first step but *before*
            # the first BFGS update to H. This is from Nocedal & Wright, 2nd. ed.,
            # Sec. 6.1 (Implementation).
            if k == 0 and H0_is_eye and self.H0_rescale_heuristic:
                H = np.dot(grad_diff, x_diff) / np.dot(grad_diff, grad_diff) * np.identity(self.obj.input_dim)
                print()
                print(f'Used H0 rescale heuristic! Scaled identity by {np.dot(grad_diff, x_diff) / np.dot(grad_diff, grad_diff)}')
                print()
                

            # Update BFGS approx. to the inverse Hessian (Nocedal & Wright,
            # 2nd ed, Eq. (6.17))
            rho = 1 / (np.dot(x_diff, grad_diff))
            H_next = (np.identity(self.obj.input_dim) - rho * np.outer(x_diff, grad_diff)) @ H @ (np.identity(self.obj.input_dim) - rho * np.outer(grad_diff, x_diff)) + rho * np.outer(x_diff, x_diff)

            # Append to recorded data
            f_vals_list.append(f_x)
            update_norms_list.append(np.linalg.norm(x_diff))
            angles_to_grad_list.append(np.arccos(np.dot(direction, - grad_vec) / (np.linalg.norm(direction) * grad_norm)) * 180 / np.pi)
            grad_norms_list.append(grad_norm)

            # Termination!
            if grad_norm < self.tol:
                x_str = ", ".join([f"{xi:7.4f}" for xi in x])
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(f"k = {k:4} || x = [{x_str}] || f(x) = {f_x:8.6e} || g_norm = {grad_norm:8.6e}")
                print('------------------------------------------------------------------------------------------')
                print()
                print()
                break
        
        f_vals = np.array(f_vals_list)
        update_norms = np.array(update_norms_list)
        angles_to_grad = np.array(angles_to_grad_list)
        grad_norms = np.array(grad_norms_list)

        return SolverOutput(solver=self,
                            final_x=x,
                            final_k=k,
                            f_vals=f_vals,
                            update_norms=update_norms,
                            angles_to_grad=angles_to_grad,
                            grad_norms=grad_norms)