""""
Here we implement a fairly general algorithm for unconstrained optimisation using line search methods.

One assumption we make is that we always use the current iterate's function value and gradient in all of these algorithms. For this reason we retain these quantities, computed at the beginning of each iteration, for further use within each iteration --- as opposed to, for example, potentially computing it twice, both in computing search direction and then in computing the step size!
"""

from autograd import grad
import autograd.numpy as np
from solvers.utils import SolverOutput

class LinesearchGeneral:
    def __init__(self, obj, deriv_info_func: callable, direction_func: callable, step_func: callable, stop_crit_func: callable, max_iter: int, iter_print_period: int=20, verbose=False, **kwargs):
        self.obj = obj # Objective class instance
        self.func = self.obj.func
        self.grad_func = grad(self.func)
        self.deriv_info_func = deriv_info_func # May return gradient, hessian, etc.
        self.direction_func = direction_func # Computes search direction
        self.step_func = step_func # Computes step size
        self.stop_crit_func = stop_crit_func # Callable returning Boolean for termination when its output at an iteration is True
        
        self.max_iter = max_iter
        self.iter_print_period = iter_print_period
        self.verbose = verbose

        # kwargs can include a variety of other variables/callables to assist in the optimisation process (stopping criteria/backtracking, etc.)
        self.kwargs = kwargs

    def optimise(self, x0: np.ndarray):
        x = x0
        k = 0
        f_x = self.func(x)

        # deriv_info should be a small list, with elements containing derivatives at x of increasing order
        deriv_info = self.deriv_info_func(x)
        
        stop_cond = self.stop_crit_func(self.obj, self.max_iter, k, x, f_x, deriv_info, self.kwargs)

        f_vals = [f_x]

        while not stop_cond:
            k += 1
            
            # Compute search direction and step
            search_dir = self.direction_func(self.obj, x, deriv_info, self.kwargs)
            step = self.step_func(self.obj, x, search_dir, deriv_info, self.kwargs) # e.g., using backtracking Armijo

            # Update relevant quantities
            x = x + step * search_dir
            
            f_x = self.obj.func(x)
            deriv_info = self.deriv_info_func(x)

            f_vals.append(f_x)
            
            stop_cond = self.stop_crit_func(self.obj, self.max_iter, k, x, f_x, deriv_info, self.kwargs)
        
        f_vals = np.array(f_vals) # convert into np array
        
        return SolverOutput(solver=self, final_x=x, final_k=k, f_vals=f_vals)

