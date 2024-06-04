"""
This script is largely based on the work from https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022).

The idea of the method proposed there is to tackle unconstrained Lipschitz-differentiable optimisation by reducing
problem to subspaces of some fixed dimension at each iterate, using previously used computation.
In an extreme example where the subspace dimension is 1, we could take linesearch methods to fall into this.
However, the idea here is to use higher subspace dimensions to speed up convergence.
"""

import autograd.numpy as np
from autograd import grad, hessian
from scipy.optimize import rosen
import matplotlib.pyplot as plt

np.random.seed(42)

class Objective:
    def __init__(self, input_dim, func):
        self.input_dim = input_dim # Input dimension
        self.func = func # callable, returns objective value

class CommonDirections:
    def __init__(self, obj, subspace_dim, reg_lambda, alpha=0.001, t_init=1, tau=0.5, tol=1e-6, max_iter=1000, verbose=False):
        """
        Initialise the optimiser with the objective function and relevant parameters.
        
        :param obj: Objective class instance.
        :param subspace_dim: Dimension of subspace.
        :param reg_lambda: Minimum allowable (POSITIVE) eigenvalue of projected Hessian. If Hessian at some point has eigenvalue below this, REGularisation will be applied to obtain a matrix with minimum eigenvalue equal to reg_lambda.
        :param alpha: The Armijo condition parameter.
        :param tau: The backtracking step size reduction factor.
        :param tol: The tolerance for the stopping condition.
        :param max_iter: The maximum number of iterations.
        """
        self.obj = obj
        self.subspace_dim = subspace_dim
        self.reg_lambda = reg_lambda
        self.func = self.obj.func # callable objective func
        self.alpha = alpha
        self.t_init = t_init
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.grad_func = grad(self.func)
        self.hess_func = hessian(self.func) # for now have it as a full Hessian; later may use autograd.hessian_vector_product
        self.verbose = verbose

    # Return regularised Hessian (or any matrix for that matter)
    # Outputs a matrix whose eigenvalue is lower-bounded by self.reg_lambda (sharp bound)
    def regularise_hessian(self, H):
        lambda_min = np.min(np.linalg.eigh(H)[0])
        if lambda_min < self.reg_lambda: # Regularise
            print('USING HESSIAN REGULARISATION!')
            H = H + (self.reg_lambda - lambda_min) * np.identity(self.subspace_dim)
        
        # TO MAKE THIS A STEEPEST DESCENT METHOD
        # H = np.identity(self.subspace_dim)
        return H

    def backtrack_armijo(self, x, direction, f_x, grad_f_x):
        """
        Perform Armijo backtracking line search to find the step size.
        
        :param x: The current point.
        :param direction: The descent direction.
        :return: The step size satisfying the Armijo condition.
        """
        t = self.t_init
        
        direction = np.squeeze(direction) # turn into vector
        while self.func(x + t * direction) > f_x + self.alpha * t * np.dot(grad_f_x, direction):
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
        f_vals = np.array([f_x])
        grad_f_x = self.grad_func(x)

        norm_grad_f_x = np.linalg.norm(grad_f_x)
        
        # Scaled Gaussian matrix (scaling unnecessary here, since we orthogonalise anyway)
        P = np.random.normal(scale=np.sqrt(1 / self.subspace_dim), size=(self.obj.input_dim, self.subspace_dim))
        P[:, -1] = np.array(grad_f_x / norm_grad_f_x) # introduce current gradient to begin with
        P, _ = np.linalg.qr(P) # orthogonalise P

        hess_f_x = self.hess_func(x)
        H = np.transpose(P) @ (hess_f_x @ P) # IN FUTURE will want to implement the latter product using Hessian actions (H-vector products), see autograd.hessian_vector_products
        H = self.regularise_hessian(H)

        # Need also to keep in parallel the raw past few gradients, in matrix G
        G = np.array(grad_f_x, ndmin=2)
        G = G.reshape(-1, 1) # column vector

        for k in range(self.max_iter):
            if norm_grad_f_x < self.tol:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(f"k = {k:4}, x = [{x_str}], f(x) = {f_x:8.6e}, g_norm = {norm_grad_f_x:8.6e}")
                break
            
            direction = - P @ np.linalg.inv(H) @ np.transpose(P) @ grad_f_x
            step_size = self.backtrack_armijo(x, direction, f_x, grad_f_x)
            
            if self.verbose and k % 20 == 0:
                x_str = ", ".join([f"{xi:7.4f}" for xi in x])
                print(f"k = {k:4}, x = [{x_str}], f(x) = {f_x:6.6e}, g_norm = {norm_grad_f_x:6.6e}, step = {step_size:8.6f}")

            # Compute new relevant quantities
            x = x + step_size * direction

            f_x = self.func(x)
            f_vals = np.append(f_vals, f_x)
            grad_f_x = self.grad_func(x)
            norm_grad_f_x = np.linalg.norm(grad_f_x)

            if G.shape[1] == self.subspace_dim:
                G = np.delete(G, 0, 1) # delete first (oldest) column
            G = np.hstack((G, grad_f_x.reshape(-1,1))) # append newest gradient
            
            if G.shape[1] == self.subspace_dim:
                P, _ = np.linalg.qr(G) # Form orthogonal basis for span of past few gradients
            else:
                P = np.hstack((G, np.random.normal(scale=np.sqrt(1 / self.subspace_dim), size=(self.obj.input_dim, self.subspace_dim - G.shape[1])))) # stack whatever gradients we already have with randomised columns
                P, _ = np.linalg.qr(P) # orthogonalise as usual

            hess_f_x = self.hess_func(x)
            H = np.transpose(P) @ (hess_f_x @ P) # IN FUTURE will want to implement the latter product using Hessian actions (H-vector products), see autograd.hessian_vector_products
            H = self.regularise_hessian(H)

        
        return x, f_vals
    
def plot_loss_vs_iteration(f_vals):
    """
    Plot the loss (function values) vs iteration count.

    :param f_vals: Array of function values over iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(f_vals, linestyle='-', color='b')
    plt.yscale('log')  # Set the vertical axis to log scale
    plt.xlabel('Iteration')
    plt.ylabel('Function value (log scale)')
    plt.title('Loss vs Iteration')
    plt.grid(True, which="both", ls="--")
    plt.show()
    
# Example usage:
if __name__ == "__main__":
    # Define the objective function

    """
    # ROSENBROCK, imported from scipy.optimize
    # Unique minimiser (f = 0) at x == np.ones(input_dim)
    input_dim = 20
    subspace_dim = 20
    x0 = np.zeros(input_dim, dtype='float32')
    func = rosen # use high-dimensional rosenbrock function from scipy.optimize
    """

    
    """
    # POWELL SINGULAR TEST FUNCTION
    input_dim = 4
    subspace_dim = 2
    x0 = np.array([3.0, -1.0, 0.0, 1.0])
    def func(x):
        return (x[0] + 10*x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10 * (x[0] - x[3])**4
    """
    
    """
    # WELL-CONDITIONED CONVEX QUADRATIC
    input_dim = 20
    subspace_dim = 20
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        out = 0
        for i in range(input_dim):
            out += (i + 1) * x[i] ** 2
        return 0.5 * out
    """


    # ILL-CONDITIONED CONVEX QUADRATIC
    input_dim = 10
    subspace_dim = 10
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        out = 0
        for i in range(input_dim):
            out += ((i + 1)**5) * x[i] ** 2
        return 0.5 * out


    # Instantiate objective class
    obj = Objective(input_dim, func)
    
    # Initialize optimiser
    optimiser = CommonDirections(obj=obj, subspace_dim=subspace_dim, alpha=0.01, reg_lambda=0.01, t_init=1, tol = 1e-3, max_iter=1000, verbose=True)

    
    # Run algorithm
    optimal_x, f_vals = optimiser.optimise(x0)

    plot_loss_vs_iteration(f_vals=f_vals)
