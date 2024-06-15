"""
This script is largely based on the work from https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022).

However, we extend that paper's algorithm with further variants including randomisation ideas.

The idea of the method proposed there is to tackle unconstrained Lipschitz-differentiable optimisation by reducing
problem to subspaces of some fixed dimension at each iterate, using previously used computation.
In an extreme example where the subspace dimension is 1, we could take linesearch methods to fall into this.
However, the idea here is to use higher subspace dimensions to speed up convergence.
"""

from dataclasses import dataclass

import autograd.numpy as np
from autograd import grad, hessian
from ..utils import SolverOutput

np.random.seed(42)

@dataclass
class CommonDirectionsConfig:
    obj: any
    subspace_update_method: str
    subspace_dim: int
    reg_lambda: float
    alpha: float = 0.001
    t_init: float = 1
    tau: float = 0.5
    tol: float = 1e-6
    max_iter: int = 1_000
    iter_print_gap:int = 20
    verbose: bool = False

class CommonDirections:
    def __init__(self, config: CommonDirectionsConfig):
        """
        Initialise the optimiser with the objective function and relevant parameters.
        
        :param obj: Objective class instance.
        :param subspace_dim: Dimension of subspace.
        :param subspace_dim: Method for building subspace basis.
        :param reg_lambda: Minimum allowable (POSITIVE) eigenvalue of projected Hessian. If Hessian at some point has eigenvalue below this, REGularisation will be applied to obtain a matrix with minimum eigenvalue equal to reg_lambda.
        :param alpha: The Armijo condition parameter.
        :param tol: The tolerance for the stopping condition.
        :param tau: The backtracking step size reduction factor.
        :param max_iter: The maximum number of iterations.
        :param iter_print_gap: Period for printing an iteration's info.
        """
        for key, value in config.__dict__.items():
            setattr(self, key, value)
        
        if self.subspace_update_method == 'iterates_grads':
            if self.subspace_dim % 2 != 0:
                raise Exception('With iterates_grads method, subspace dimension must be multiple of 2.')
        if self.subspace_update_method == 'iterates_grads_diagnewtons':
            if self.subspace_dim % 3 != 0:
                raise Exception('With iterates_grads_diagnewtons method, subspace dimension must be multiple of 3.')
        
        self.func = self.obj.func # callable objective func
        self.grad_func = grad(self.func)
        self.hess_func = hessian(self.func) # for now have it as a full Hessian; later may use autograd.hessian_vector_product

    # Return regularised Hessian (or any matrix for that matter)
    # Outputs a matrix whose eigenvalue is lower-bounded by self.reg_lambda (sharp bound)
    def regularise_hessian(self, H):
        lambda_min = np.min(np.linalg.eigh(H)[0])
        if lambda_min < self.reg_lambda: # Regularise
            # print('USING HESSIAN REGULARISATION!')
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
    
    # Which basis of subspace to use in the method
    def update_subspace(self, method: str, **kwargs):
        # method: str. Options:
        # method == 'grads', retain m past gradient vectors 
        # method == 'iterates_grads', (34) from Lee et al., 2022
        # method == 'iterates_grads_diagnewtons', (35) from Lee et al., 2022

        # G should store some past gradient vectors
        # X should store some past iterate vectors
        # D should store some past crude Newton direction approximations (obtained by multiplying the inverse of the diagonal of Hessian diagonal by the gradient)
        if self.subspace_update_method == 'grads':
            G = kwargs['grads_matrix']
            if G.shape[1] == self.subspace_dim:
                P = G
            else:
                P = np.zeros(shape=(self.obj.input_dim, self.subspace_dim))
                P[:,:(self.subspace_dim - G.shape[1])] = np.random.normal(size=(self.obj.input_dim, self.subspace_dim - G.shape[1]))
                P[:,-G.shape[1]:] = G
        elif self.subspace_update_method == 'iterates_grads':
            G = kwargs['grads_matrix']
            X = kwargs['iterates_matrix']
            if 2 * G.shape[1] == self.subspace_dim:
                P = np.hstack((G, X))
            else:
                P = np.zeros(shape=(self.obj.input_dim, self.subspace_dim))
                P[:,:(self.subspace_dim - 2 * G.shape[1])] = np.random.normal(size=(self.obj.input_dim, self.subspace_dim - 2 * G.shape[1]))
                P[:,-2 * G.shape[1]:] = np.hstack((G, X))
        elif self.subspace_update_method == 'iterates_grads_diagnewtons':
            G = kwargs['grads_matrix']
            X = kwargs['iterates_matrix']
            D = kwargs['hess_diag_dirs_matrix']
            if 3 * G.shape[1] == self.subspace_dim:
                P = np.hstack((G, X, D))
            else:
                P = np.zeros(shape=(self.obj.input_dim, self.subspace_dim))
                P[:,:(self.subspace_dim - 3 * G.shape[1])] = np.random.normal(size=(self.obj.input_dim, self.subspace_dim - 3 * G.shape[1]))
                P[:,-3 * G.shape[1]:] = np.hstack((G, X, D))
        
        # Orthogonalise the basis matrix
        print(f'Cond. number of basis matrix pre-QR: {np.linalg.cond(P):8.6e}')
        P, _ = np.linalg.qr(P)
        return P

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
        hess_f_x = self.hess_func(x)

        # For further plotting
        f_vals = np.array([f_x], dtype='float32')

        # Need to keep in parallel information from few previous iterates
        G = np.array(grad_f_x, ndmin=2) # store gradient vectors
        G = G.reshape(-1, 1)
        if self.subspace_update_method != 'grads':
            X = np.array(x, ndmin=2) # store iterates
            X = X.reshape(-1, 1)
        else:
            X = None
        if self.subspace_update_method == 'iterates_grads_diagnewtons':
            D = np.linalg.solve(np.diag(np.diag(hess_f_x)), grad_f_x)
            D = D.reshape(-1, 1)
        else:
            D = None
        
        P = self.update_subspace(self.subspace_update_method, grads_matrix=G, iterates_matrix=X, hess_diag_dirs_matrix=D)

        H = np.transpose(P) @ (hess_f_x @ P) # IN FUTURE will want to implement the latter product using Hessian actions (H-vector products), see autograd.hessian_vector_products
        H = self.regularise_hessian(H)

        for k in range(self.max_iter):
            if norm_grad_f_x < self.tol:
                x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(f"k = {k:4}, x = [{x_str}], f(x) = {f_x:8.6e}, g_norm = {norm_grad_f_x:8.6e}")
                print('------------------------------------------------------------------------------------------')
                print()
                print()
                break
            
            direction = - P @ np.linalg.inv(H) @ np.transpose(P) @ grad_f_x
            step_size = self.backtrack_armijo(x, direction, f_x, grad_f_x)
            
            if self.verbose and k % self.iter_print_gap == 0:
                x_str = ", ".join([f"{xi:7.4f}" for xi in x])
                print(f"k = {k:4}, x = [{x_str}], f(x) = {f_x:6.6e}, g_norm = {norm_grad_f_x:6.6e}, step = {step_size:8.6f}")

            # Compute new relevant quantities
            x = x + step_size * direction

            f_x = self.func(x)
            grad_f_x = self.grad_func(x)
            hess_f_x = self.hess_func(x)
            norm_grad_f_x = np.linalg.norm(grad_f_x)
            
            # Update records of previous iterations, for further plotting
            f_vals = np.append(f_vals, f_x)

            if self.subspace_update_method == 'grads':
                if G.shape[1] == self.subspace_dim:
                    G = np.delete(G, 0, 1) # delete first (oldest) column
                G = np.hstack((G, grad_f_x.reshape(-1,1))) # append newest gradient
            elif self.subspace_update_method == 'iterates_grads':
                if 2 * G.shape[1] == self.subspace_dim:
                    G = np.delete(G, 0, 1) # delete first (oldest) column
                    X = np.delete(X, 0, 1)
                G = np.hstack((G, grad_f_x.reshape(-1,1))) # append newest gradient
                X = np.hstack((X, x.reshape(-1,1))) # append newest iterate
            elif self.subspace_update_method == 'iterates_grads_diagnewtons':
                if 3 * G.shape[1] == self.subspace_dim:
                    G = np.delete(G, 0, 1) # delete first (oldest) column
                    X = np.delete(X, 0, 1)
                    D = np.delete(D, 0, 1)
                G = np.hstack((G, grad_f_x.reshape(-1,1))) # append newest gradient
                X = np.hstack((X, x.reshape(-1,1))) # append newest iterate
                D = np.hstack((D, np.linalg.solve(np.diag(np.diag(hess_f_x)), grad_f_x).reshape(-1, 1))) # append newest crude diagonal Newton direction approximation
            
            P = self.update_subspace(self.subspace_update_method, grads_matrix=G, iterates_matrix=X, hess_diag_dirs_matrix=D)

            H = np.transpose(P) @ (hess_f_x @ P) # IN FUTURE will want to implement the latter product using Hessian actions (H-vector products), see autograd.hessian_vector_products
            H = self.regularise_hessian(H)

        
        return SolverOutput(solver=self, final_x=x, final_k=k, f_vals=f_vals)
