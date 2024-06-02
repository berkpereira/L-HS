"""
This script is largely based on the work from https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022).

The idea of the method proposed there is to tackle unconstrained Lipschitz-differentiable optimisation by reducing
problem to subspaces of some fixed dimension at each iterate, using previously used computation.
In an extreme example where the subspace dimension is 1, we could take linesearch methods to fall into this.
However, the idea here is to use higher subspace dimensions to speed up convergence.
"""

import autograd.numpy as np
from autograd import grad

class Objective:
    def __init__(self, input_dim, func):
        self.input_dim = input_dim # Input dimension
        self.func = func # callable, returns objective value

class CommonDirections:
    def __init__(self, obj, order, subspace_dim, alpha=0.001, t_init=1, tau=0.5, tol=1e-6, max_iter=1000, verbose=False):
        """
        Initialise the optimiser with the objective function and relevant parameters.
        
        :param obj: Objective class instance.
        :param order: int in {1, 2}: Order of local model at each iterate.
        :param subspace_dim: Dimension of subspace.
        :param alpha: The Armijo condition parameter.
        :param tau: The backtracking step size reduction factor.
        :param tol: The tolerance for the stopping condition.
        :param max_iter: The maximum number of iterations.
        """
        self.obj = obj
        self.order = order
        self.subspace_dim = subspace_dim
        self.func = self.obj.func # callable objective func
        self.alpha = alpha
        self.t_init = t_init
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.grad_func = grad(self.func)
        self.verbose = verbose

    def backtrack_armijo(self, x, direction, f_x, grad_f_x):
        """
        Perform Armijo backtracking line search to find the step size.
        
        :param x: The current point.
        :param direction: The descent direction.
        :return: The step size satisfying the Armijo condition.
        """
        t = self.t_init
        
        direction = np.squeeze(direction) # turn into vector
        while self.func(x + t * direction) > f_x + self.alpha * t * np.dot(np.transpose(grad_f_x), direction):
            t *= self.tau
        
        return t
    
    def optimise(self, x0):
        """
        Perform solver step
        
        :param x0: The initial guess for the minimum.
        :return: The point that approximately minimizes the objective function.
        """
        # Initialise algorithm
        x = x0
        f_x = self.func(x)
        grad_f_x = self.grad_func(x)
        norm_grad_f_x = np.linalg.norm(grad_f_x)
        P = np.array(grad_f_x / norm_grad_f_x, ndmin=2) # normalise
        P = P.reshape(-1, 1) # column vector

        for k in range(self.max_iter):
            if norm_grad_f_x < self.tol:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}")
                break
            
            direction = -grad_f_x
            step_size = self.backtrack_armijo(x, direction, f_x, grad_f_x)
            
            if self.verbose and k % 1000 == 0:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}, step size = {step_size:8.6f}")

            # New iterate
            x = x + step_size * direction
            
            f_x = self.func(x)
            grad_f_x = self.grad_func(x)            
            norm_grad_f_x = np.linalg.norm(grad_f_x)
            
            q_new = grad_f_x.reshape(-1, 1) # use as a column vector for updating P
            if P.shape[1] == self.subspace_dim:
                P = np.delete(P, 0, 1) # delete first (oldest) column
            q_new = q_new - (P @ np.transpose(P) @ q_new) # Gram-Schmidt orthogonalisation
            q_new = q_new / np.linalg.norm(q_new)
            P = np.hstack((P, q_new))

        
        return x
    
# Example usage:
if __name__ == "__main__":
    # Define the objective function
    """
    # ROSENBROCK
    input_dim = 2
    x0 = np.array([0.0, 0.0])
    def func(x):
        a, b = 1, 100
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    """
    
    # POWELL SINGULAR TEST FUNCTION
    input_dim = 4
    x0 = np.array([3.0, -1.0, 0.0, 1.0])
    def func(x):
        return (x[0] + 10*x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10 * (x[0] - x[3])**4
    
    # Instantiate objective class
    obj = Objective(input_dim, func)
    
    # Initialize optimiser
    optimiser = CommonDirections(obj=obj, order=1, subspace_dim=3, t_init=0.1, tol = 1e-5, max_iter=100000, verbose=True)

    
    # Run algorithm
    optimal_x = optimiser.optimise(x0)