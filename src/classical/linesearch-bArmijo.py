import autograd.numpy as np
from autograd import grad, hessian
import matplotlib.pyplot as plt
from scipy.optimize import rosen

class Objective:
    def __init__(self, input_dim, func):
        self.input_dim = input_dim # Input dimension
        self.func = func # callable, returns objective value

class LinesearchBacktrackingArmijo:
    def __init__(self, method, obj, alpha=0.001, t_init=1, tau=0.5, tol=1e-6, max_iter=1000, verbose=False):
        """
        Initialise the optimiser with the objective function and parameters.
        
        :param method: String choosing method for linesearch direction.
        :param func: The objective function to minimize.
        :param alpha: The Armijo condition parameter.
        :param tau: The backtracking step size reduction factor.
        :param tol: The tolerance for the stopping condition.
        :param max_iter: The maximum number of iterations.
        """
        self.method = method
        self.obj = obj
        self.func = self.obj.func
        self.alpha = alpha
        self.t_init = t_init
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.grad_func = grad(func)
        self.hess_func = hessian(self.func)
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
            step_size = self.backtrack_armijo(x, direction, f_x, grad_f_x)
            
            if self.verbose and k % 1 == 0:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {norm_grad_f_x:10.6e}, step size = {step_size:8.6f}")

            
            x = x + step_size * direction
        
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
    # METHOD EITHER 'SD' or 'Newton'
    METHOD = 'Newton'

    """
    # ROSENBROCK, imported from scipy.optimize
    # Unique minimiser (f = 0) at x == np.ones(input_dim)
    input_dim = 20
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
    x0 = np.ones(input_dim, dtype='float32')
    def func(x):
        out = 0
        for i in range(input_dim):
            out += ((i + 1)**5) * x[i] ** 2
        return 0.5 * out
    
    # Instantiate objective class
    obj = Objective(input_dim, func)
    
    # Initialize optimiser
    optimiser = LinesearchBacktrackingArmijo(method=METHOD, obj=obj, alpha=0.01, t_init=1, tol = 1e-3, max_iter=1000, verbose=True)
    
    # Perform optimisation
    optimal_x, f_vals = optimiser.optimise(x0)

    plot_loss_vs_iteration(f_vals=f_vals)
