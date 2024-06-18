"""
In this module we implement quasi-Newton methods, in particular of the unlimited-memory sort (i.e., we do not strive to save up on storage of secant pairs, etc., as one may do with a method like L-BFGS).

Throughout, we denote, at iteration k, an approximation to the Hessian by B_k, and the inverse of B_k by H_k. Using Sherman-Morrison-Woodbury formula, we can directly update the inverse H_k as opposed to B_k, sidestepping the solution of a linear system in determining the search direction.

The BFGS implementation follows the description from the well-known textbook Numerical Optimization by Nocedal and Wright, 2nd ed., Chapter 6.1.










TODO:
Implement a linesearch method ensuring the Wolfe conditions!
Look at Nocedal & Wright, 2nd ed. page 60, which is also implemented in scipy.optimize.line_search !

Then try out the method!
(Then implement L-BFGS...)








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
        # Initialise with initial inputs given
        x = x0
        H = H0
        
        x_next = x
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
            step_size = self.wolfe_linesearch(x, ...)

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

            # Update approx. to the inverse Hessian (Nocedal & Wright, 2nd ed, Eq. (6.17))
            rho = 1 / (np.dot(x_diff, grad_diff))
            H_next = (np.identity(self.obj.input_dim) - rho * np.outer(x_diff, grad_diff)) @ H @ (np.identity(self.obj.input_dim) - rho * np.outer(grad_diff, x_diff)) + rho * np.outer(x_diff, x_diff)


            # Append to recorded data
            f_vals_list.append(f_x)
            update_norms_list.append()
            angles_to_grad_list.append()
            grad_norms_list.append()

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