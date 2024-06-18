"""
This script is largely based on the work from https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022). There is another module in this repo, common_directions.py, reflecting the algorithms proposed in that paper closely. This module adapts that with some changes.

Namely, here we implement a version of that algorithm that never uses full gradient information.
We also seek to make this version of the algorithm more general, including variants that do/do not use Hessian information, etc.
The common thread to all of these methods is that THE FULL GRADIENT IS NEVER USED IN ITSELF IN THE ALGORITHM.

For discussion of these ideas, see relevant docs/*.md file.
"""

from dataclasses import dataclass

import autograd.numpy as np
from autograd import grad, hessian
from utils import SolverOutput

np.random.seed(42)

@dataclass
class ProjectedCommonDirectionsConfig:
    obj: any                    # Objective class instance.
    subspace_update_method: str # Determines method for updating subspace.
    subspace_dim: int           # Dimension of subproblem subspace.
    reg_lambda: float           # Minimum allowable (POSITIVE) eigenvalue of projected Hessian.
    use_hess: bool = True       # Determines whether method uses Hessian information. If not, it uses in general a user-specified matrix B_k (think quasi-Newton methods). Default of True reflects Lee et al.'s reference paper's method.
    random_proj: bool = False   # Determines whether to use RANDOM matrices in projecting gradients.
    ensemble: str = ''          # Determines the random ensemble from which to draw random matrices for gradient projections.
    alpha: float = 0.001        # The Armijo condition scaling parameter.
    t_init: float = 1
    tau: float = 0.5            # The backtracking step size reduction factor.
    tol: float = 1e-6           # The tolerance for the stopping condition.
    max_iter: int = 1_000       # The maximum number of iterations.
    iter_print_gap:int = 20     # Period for printing an iteration's info.
    verbose: bool = False

class ProjectedCommonDirections:
    def __init__(self, config: ProjectedCommonDirectionsConfig):
        # Set all attributes given in ProjectedCommonDirectionsConfig
        for key, value in config.__dict__.items():
            setattr(self, key, value)
        
        # Some simple checks on allowable subspace dimension
        if self.subspace_update_method == 'iterates_grads':
            if self.subspace_dim % 2 != 0:
                raise Exception('With iterates_grads method, subspace dimension must be multiple of 2.')
        if self.subspace_update_method == 'iterates_grads_diagnewtons':
            if self.subspace_dim % 3 != 0:
                raise Exception('With iterates_grads_diagnewtons method, subspace dimension must be multiple of 3.')
        
        self.func = self.obj.func # callable objective func
        self.grad_func = grad(self.func)
        self.hess_func = hessian(self.func) # for now have it as a full Hessian; later may use autograd.hessian_vector_product

    # Draw a TALL sketching matrix from random ensemble.
    def draw_sketch(self):
        # Recovering full-dimensional method.
        if self.subspace_dim == self.obj.input_dim:
            return np.identity(self.obj.input_dim)
        elif self.ensemble == 'scaled_gaussian':
            return np.random.normal(scale=np.sqrt(1 / self.subspace_dim), size=(self.obj.input_dim, self.subspace_dim))
        else:
            raise Exception('Unrecognised sketching matrix scheme!')

    # Return regularised Hessian (or any matrix for that matter)
    # Outputs a matrix whose eigenvalue is lower-bounded by self.reg_lambda (sharp bound)
    def regularise_hessian(self, B):
        lambda_min = np.min(np.linalg.eigh(B)[0])
        if lambda_min < self.reg_lambda: # Regularise
            print('USING HESSIAN REGULARISATION!')
            B = B + (self.reg_lambda - lambda_min) * np.identity(self.subspace_dim)
            
        return B

    # This method takes in the current iterate and performs a backtracking Armijo linesearch along the specified search direction until the Armijo condition is satisfied (self attributes used where appropriate).
    def backtrack_armijo(self, x, direction, f_x, grad_vec):
        t = self.t_init
        
        direction = np.squeeze(direction) # turn into vector
        while self.func(x + t * direction) > f_x + self.alpha * t * np.dot(grad_vec, direction):
            t *= self.tau
        
        return t
    
    # The below method is new compared to the common_directions.py module.
    # It plays the role of determining how "projected gradients are defined".
    def project_gradient(self, full_grad, random_proj, **kwargs):
        if random_proj:
            W = self.draw_sketch()
        else:
            W = kwargs['Q_prev']
        proj_grad = W @ np.transpose(W) @ full_grad
        return proj_grad

    # Which basis of subspace to use in the method
    def update_subspace(self, **kwargs):
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
        Q, _ = np.linalg.qr(P)
        return Q, np.linalg.cond(P)

    def optimise(self, x0):
        # Initialise algorithm
        x = x0
        f_x = self.func(x)
        full_grad = self.grad_func(x)
        norm_full_grad = np.linalg.norm(full_grad)

        # The first projected gradient takes a projection from an entirely random matrix
        proj_grad = self.project_gradient(full_grad, random_proj=True)

        # IN FUTURE will want to implement Hessian actions product using Hessian actions (B-vector products), see autograd.hessian_vector_products
        if self.use_hess:
            hess_f_x = self.hess_func(x)
            full_B = hess_f_x
        else:
            raise Exception('Have not (yet!) implemented methods with user-provided B approximations to the Hessian!')

        # For later plotting
        f_vals_list = [f_x]
        update_norms_list = []
        angles_to_full_grad_list = []
        full_grad_norms_list = [np.linalg.norm(full_grad)]
        proj_grad_norms_list = [np.linalg.norm(proj_grad)]

        # Need to keep in parallel information from few previous iterates.
        # Initialise this here, depending on subspace method used:
        G = np.array(proj_grad, ndmin=2) # store gradient vectors
        G = G.reshape(-1, 1)

        # G_proj_unlimited = G # for storage of ALL PROJECTED gradients (for debugging)
        # G_full_unlimited = np.array(full_grad, ndmin=2)
        # G_full_unlimited = G_full_unlimited.reshape(-1, 1)

        if self.subspace_update_method != 'grads':
            X = np.array(x, ndmin=2) # store iterates
            X = X.reshape(-1, 1)
        else:
            X = None
        if self.subspace_update_method == 'iterates_grads_diagnewtons':
            D = np.linalg.solve(np.diag(np.diag(hess_f_x)), proj_grad)
            D = D.reshape(-1, 1)
        else:
            D = None
        
        Q, last_cond_no = self.update_subspace(grads_matrix=G, iterates_matrix=X, hess_diag_dirs_matrix=D)

        cond_nos_list = [last_cond_no]

        # Project B matrix
        # Later may want to do this using Hessian actions in the case where Hessian information is used at all.
        proj_B = np.transpose(Q) @ (full_B @ Q)

        # Regularise the projected/reduced Hessian approximation if needed.
        proj_B = self.regularise_hessian(proj_B)

        for k in range(self.max_iter):
            if norm_full_grad < self.tol:
                x_str = ", ".join([f"{xi:7.4f}" for xi in x])
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(f"k = {k:4} || x = [{x_str}] || f(x) = {f_x:8.6e} || g_norm = {norm_full_grad:8.6e}")
                print('------------------------------------------------------------------------------------------')
                print()
                print()
                break
            
            direction = - Q @ np.linalg.inv(proj_B) @ np.transpose(Q) @ proj_grad
            step_size = self.backtrack_armijo(x, direction, f_x, proj_grad)
            
            if self.verbose and k % self.iter_print_gap == 0:
                x_str = ", ".join([f"{xi:7.4f}" for xi in x])
                print(f"k = {k:4} || x = [{x_str}] || f(x) = {f_x:6.6e} || g_norm = {norm_full_grad:6.6e} || t = {step_size:8.6f}")
            
            
            x = x + step_size * direction


            last_update_norm = np.linalg.norm(step_size * direction)
            last_angle_to_full_grad = np.dot(direction, -full_grad) / (np.linalg.norm(direction) * np.linalg.norm(full_grad)) * 180 / np.pi
            angles_to_full_grad_list.append(last_angle_to_full_grad)

            update_norms_list.append(last_update_norm)

            if self.random_proj and last_update_norm < 1e-10:
                print('Very small update with randomised projections!')


            f_x = self.func(x)
            full_grad = self.grad_func(x)
            norm_full_grad = np.linalg.norm(full_grad)

            proj_grad = self.project_gradient(full_grad, random_proj=self.random_proj, Q_prev=Q)


            full_grad_norms_list.append(np.linalg.norm(full_grad))
            proj_grad_norms_list.append(np.linalg.norm(proj_grad))


            # G_proj_unlimited = np.hstack((G_proj_unlimited, proj_grad.reshape(-1,1)))
            # G_full_unlimited = np.hstack((G_full_unlimited, full_grad.reshape(-1,1)))

            if self.use_hess:
                hess_f_x = self.hess_func(x)
                full_B = hess_f_x
            else:
                raise Exception('Not yet implemented!')
            
            # Update records of previous iterations, for further plotting
            f_vals_list.append(f_x)

            if self.subspace_update_method == 'grads':
                if G.shape[1] == self.subspace_dim:
                    G = np.delete(G, 0, 1) # delete first (oldest) column
                G = np.hstack((G, proj_grad.reshape(-1,1))) # append newest gradient
            elif self.subspace_update_method == 'iterates_grads':
                if 2 * G.shape[1] == self.subspace_dim:
                    G = np.delete(G, 0, 1) # delete first (oldest) column
                    X = np.delete(X, 0, 1)
                G = np.hstack((G, proj_grad.reshape(-1,1))) # append newest gradient
                X = np.hstack((X, x.reshape(-1,1))) # append newest iterate
            elif self.subspace_update_method == 'iterates_grads_diagnewtons':
                raise Exception('Implementation of "iterates_grads_diagnewtons" subspace construction in the projected case not yet thought out!')
                if 3 * G.shape[1] == self.subspace_dim:
                    G = np.delete(G, 0, 1) # delete first (oldest) column
                    X = np.delete(X, 0, 1)
                    D = np.delete(D, 0, 1)
                G = np.hstack((G, proj_grad.reshape(-1,1))) # append newest gradient
                X = np.hstack((X, x.reshape(-1,1))) # append newest iterate
                D = np.hstack((D, np.linalg.solve(np.diag(np.diag(hess_f_x)), proj_grad).reshape(-1, 1))) # append newest crude diagonal Newton direction approximation
            
            Q, last_cond_no = self.update_subspace(grads_matrix=G, iterates_matrix=X, hess_diag_dirs_matrix=D)

            cond_nos_list.append(last_cond_no)

            proj_B = np.transpose(Q) @ (full_B @ Q)
            proj_B = self.regularise_hessian(proj_B)

        # Convert these to arrays for later plotting
        f_vals = np.array(f_vals_list)
        update_norms = np.array(update_norms_list)
        full_grad_norms = np.array(full_grad_norms_list)
        proj_grad_norms = np.array(proj_grad_norms_list)
        angles_to_full_grad = np.array(angles_to_full_grad_list)
        cond_nos = np.array(cond_nos_list) # condition numbers of P matrix at each iteration

        return SolverOutput(solver=self, final_x=x, final_k=k, f_vals=f_vals, 
                            update_norms=update_norms,
                            full_grad_norms=full_grad_norms,
                            proj_grad_norms=proj_grad_norms,
                            angles_to_full_grad=angles_to_full_grad,
                            cond_nos=cond_nos)
