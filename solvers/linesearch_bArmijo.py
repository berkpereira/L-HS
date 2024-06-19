import autograd.numpy as np
from autograd import grad, hessian
from solvers.utils import SolverOutput

class LinesearchBacktrackingArmijo:
    def __init__(self, method, obj, alpha=0.001, t_init=1, tau=0.5, tol=1e-6, max_iter=1000, iter_print_gap=20, verbose=False):
        """
        Initialise the optimiser with the objective function and parameters.
        
        :param method: String choosing method for linesearch direction, one of {'SD', 'Newton'}
        :param func: The objective function to minimize.
        :param alpha: The Armijo condition parameter.
        :param tau: The backtracking step size reduction factor.
        :param tol: The tolerance for the stopping condition.
        :param max_iter: The maximum number of iterations.
        :param iter_print_gap: "Period" (in iterations) for printing an iteration's info.
        """
        self.method = method
        self.obj = obj
        self.func = self.obj.func
        self.alpha = alpha
        self.t_init = t_init
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.grad_func = grad(self.func)
        self.hess_func = hessian(self.func)
        self.iter_print_gap = iter_print_gap
        self.verbose = verbose

    def backtrack_armijo(self, x, direction, f_x, grad_f_x):
        """
        Perform Armijo backtracking line search to find the step size.
        
        :param x: The current point.
        :param direction: The descent direction.
        :return: The step size satisfying the Armijo condition.
        """
        t = self.t_init
        
        while self.func(x + t * direction) > f_x + self.alpha * t * np.dot(grad_f_x, direction):
            t *= self.tau
        
        return t
    
    def optimise(self, x0):
        """
        Perform the steepest descent optimisation.
        
        :param x0: The initial guess for the minimum.
        :return: The point that approximately minimizes the objective function.
        """
        x = x0
        for k in range(self.max_iter):
            f_x = self.func(x)
            if k == 0:
                f_vals = np.array([f_x])
            else:
                f_vals = np.append(f_vals, f_x)
            grad_f_x = self.grad_func(x)
            norm_grad_f_x = np.linalg.norm(grad_f_x)
            
            if norm_grad_f_x < self.tol:
                if self.verbose:
                    x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                    print('------------------------------------------------------------------------------------------')
                    print('TERMINATED')
                    print('------------------------------------------------------------------------------------------')
                    print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}")
                break
            
            if self.method == 'SD':
                direction = -grad_f_x
            elif self.method == 'Newton':
                H = self.hess_func(x)
                direction = - np.linalg.inv(H) @ grad_f_x # Newton direction
            else:
                raise Exception('Unkown method!')
            step_size = self.backtrack_armijo(x, direction, f_x, grad_f_x)
            
            if self.verbose and k % self.iter_print_gap == 0:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}, step size = {step_size:8.6f}")

            
            x = x + step_size * direction
        
        return SolverOutput(x, k, f_vals)