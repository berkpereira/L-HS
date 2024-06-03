import autograd.numpy as np
from autograd import grad

class SteepestDescentBacktrackingArmijo:
    def __init__(self, func, alpha=0.001, t_init=1, tau=0.5, tol=1e-6, max_iter=1000, verbose=False):
        """
        Initialise the optimiser with the objective function and parameters.
        
        :param func: The objective function to minimize.
        :param alpha: The Armijo condition parameter.
        :param tau: The backtracking step size reduction factor.
        :param tol: The tolerance for the stopping condition.
        :param max_iter: The maximum number of iterations.
        """
        self.func = func
        self.alpha = alpha
        self.t_init = t_init
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.grad_func = grad(func)
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
            grad_f_x = self.grad_func(x)
            norm_grad_f_x = np.linalg.norm(grad_f_x)
            
            if norm_grad_f_x < self.tol:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}")
                break
            
            direction = -grad_f_x
            step_size = self.backtrack_armijo(x, direction, f_x, grad_f_x)
            
            if self.verbose:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}, step size = {step_size:8.6f}")

            
            x = x + step_size * direction
        
        return x

# Example usage:
if __name__ == "__main__":
    # Define the objective function
    def func(x):
        a, b = 1, 100
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    # Initialize optimiser
    optimiser = SteepestDescentBacktrackingArmijo(func, t_init=5, tol = 1e-5, max_iter=10000, verbose=True)
    
    # Initial guess
    x0 = np.array([0.0, 0.0])
    
    # Perform optimisation
    optimal_x = optimiser.optimise(x0)
