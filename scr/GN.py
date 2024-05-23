# Classical Gauss-Newton method

import autograd.numpy as np
from autograd import jacobian

class GaussNewtonOptimiser:
    def __init__(self, res, alpha=0.001, t_init=1, tau=0.5, tol=1e-6, max_iter=1000, verbose=False):
        """
        Initialize the optimiser with the objective function and parameters.

        :param res: The residual vetor function for the nonlinear least squares problem.
        :param tol: The tolerance for the stopping condition.
        :param max_iter: The maximum number of iterations.
        :param verbose: If True, print detailed iteration information.
        """
        self.res = res
        self.func = lambda x: 0.5 * np.dot(self.res(x),self.res(x))
        self.alpha = alpha
        self.t_init = t_init
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.jacobian_func = jacobian(res)

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
        Perform Gauss-Newton optimisation with Armijo linesearch.

        :param x0: The initial guess for the minimum.
        :return: The point that approximately minimizes the objective function.
        """
        x = x0
        for k in range(self.max_iter):
            res = self.res(x) # residual vector
            J = self.jacobian_func(x)
            grad_f_x = J.T @ res
            hess_tilde_f_x = J.T @ J
            norm_grad_f_x = np.linalg.norm(grad_f_x)

            if norm_grad_f_x < self.tol:
                f_x = self.func(x)
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}")
                break

            if self.verbose:
                f_x = self.func(x)
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}")

            direction = np.linalg.solve(hess_tilde_f_x, -grad_f_x)
            step_size = self.backtrack_armijo(x, direction, f_x, grad_f_x)

            x = x + step_size * direction

        return x

# Example usage:
if __name__ == "__main__":
    # Define the residual function for a simple nonlinear least squares problem
    def residual(x):
        return np.array([
            10 * (x[1] - x[0]**2),
            1 - x[0]
        ])
    
    # Initialize optimiser
    optimiser = GaussNewtonOptimiser(residual, tol=1e-6, max_iter=100, verbose=True)
    
    # Initial guess
    x0 = np.array([20, -20], dtype='float32')
    
    # Perform optimisation
    optimal_x = optimiser.optimise(x0)
    
    print("Optimal x:", optimal_x)
