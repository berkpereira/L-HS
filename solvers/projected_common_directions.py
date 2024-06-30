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
from solvers.utils import SolverOutput, scaled_gaussian, haar, append_orth_dirs

@dataclass
class ProjectedCommonDirectionsConfig:
    """
    obj: Objective class instance.
    subspace_constr_method: Determines method for updating subspace.
    subspace_dim: Dimension of subproblem subspace.
    reg_lambda: Minimum allowable (POSITIVE) eigenvalue of projected Hessian.
    ...
    
    reproject_grad: NOTE: relevant only to DETERMINISTIC variant. If False, each
    gradient is projected only once, using the Q matrix from the previous iter.
    If True, then each gradient is also reprojected, at the next iteration,
    using its own iteration's Q matrix.

    append_rand_dirs: NOTE: Determines how many random directions to append
    to the (otherwise deterministic) subspace matrix at each iteration.
    Uses solvers.utils.append_orth_dirs.
    
    """
    obj: any                    
    # subspace_constr_method: str
    # subspace_dim: int
    # append_rand_dirs: int = 0
    reg_lambda: float
    
    subspace_no_grads: int = 0
    subspace_no_updates: int = 0
    subspace_no_random: int = 0

    random_proj_dim: int = 0 # Dimension of random sketches used when random_proj is True

    use_hess: bool = True       # Determines whether method uses Hessian information. If not, it uses in general a user-specified matrix B_k (think quasi-Newton methods). Default of True reflects Lee et al.'s reference paper's method.
    random_proj: bool = False   # Determines whether to use RANDOM matrices in projecting gradients for subspace construction.
    reproject_grad: bool = False
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

        # Overall number of columns in P_k
        self.subspace_dim = self.subspace_no_grads + self.subspace_no_updates + self.subspace_no_random

        # Distinguish case when no problem information is used in P_k, i.e.,
        # where P_k is fully random.
        if self.subspace_no_grads == 0 and self.subspace_no_updates == 0:
            self.subspace_constr_method = 'random'
        else:
            self.subspace_constr_method = None
        
        # Number of P_k columns relying on problem/algorithm information
        # of some kind. 
        self.no_problem_dirs = self.subspace_no_grads + self.subspace_no_updates

        # if self.subspace_constr_method == 'random' and self.append_rand_dirs > 0:
        #     raise Exception('It makes no sense to append random directions when the entire subspace construction is random anyway!')

        if self.no_problem_dirs <= 0 and self.subspace_constr_method != 'random':
            raise Exception('It makes no sense to have no problem information in subspace construction when not using a purely randomised subspace approach!')
        
        # Some simple checks on allowable subspace dimension
        # if self.subspace_constr_method == 'iterates_grads':
        #     if self.no_problem_dirs % 2 != 0:
        #         raise Exception('With iterates_grads method, "deterministic" subspace dimension must be multiple of 2.')
        # if self.subspace_constr_method == 'iterates_grads_diagnewtons':
        #     if self.no_problem_dirs % 3 != 0:
        #         raise Exception('With iterates_grads_diagnewtons method, "deterministic" subspace dimension must be multiple of 3.')
        
        # Checks on gradient reprojection and whether subspace dimension allows it
        if self.reproject_grad and (not self.random_proj) and self.subspace_no_grads <= 1:
            raise Exception("Gradients can only be reprojected if we're storing more than one for each subspace!")
        
        self.func = self.obj.func # callable objective func
        self.grad_func = grad(self.func)
        self.hess_func = hessian(self.func) # for now have it as a full Hessian; later may use autograd.hessian_vector_product

    # Draw a TALL sketching matrix from random ensemble.
    def draw_sketch(self):
        if self.random_proj_dim == self.obj.input_dim: # Recovering full-dimensional method
            return np.identity(self.obj.input_dim)
        elif self.ensemble == 'scaled_gaussian':
            return scaled_gaussian(self.obj.input_dim, self.random_proj_dim)
        elif self.ensemble == 'haar':
            return haar(self.obj.input_dim, self.random_proj_dim)
        else:
            raise Exception('Unrecognised sketching matrix scheme!')

    # Return regularised Hessian (or any matrix for that matter)
    # Outputs a matrix whose eigenvalue is lower-bounded by self.reg_lambda (sharp bound)
    def regularise_hessian(self, B):
        lambda_min = np.min(np.linalg.eigh(B)[0])
        if lambda_min < self.reg_lambda: # Regularise
            print('USING HESSIAN REGULARISATION!') # Notify
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
    # It plays the role of determining how "projected gradients" are defined.
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

        # G should store some past PROJECTED gradient vectors.
        # X should store some past iterate update vectors.
        # D should store some past crude Newton direction
        # approximations (obtained by multiplying the inverse of the
        # diagonal of Hessian diagonal by the gradient).

        if self.subspace_constr_method == 'random':
            P = None
            
            # Add orthogonal random directions as necessary
            Q = append_orth_dirs(curr_mat=P,
                                ambient_dim=self.obj.input_dim,
                                no_dirs=self.subspace_dim,
                                curr_is_orth=False)
            return Q, None, None
        else:
            G = kwargs['grads_matrix']
            X = kwargs['updates_matrix']
            D = kwargs['hess_diag_dirs_matrix']
            
            arrays = [arr for arr in [G, X, D] if arr is not None]
            P = np.hstack(arrays)
            
            # Add orthogonal random directions as necessary
            Q = append_orth_dirs(curr_mat=P,
                                ambient_dim=self.obj.input_dim,
                                no_dirs=self.subspace_dim - P.shape[1],
                                curr_is_orth=False)
            return Q, np.linalg.cond(P), np.linalg.matrix_rank(P)

        # if self.subspace_constr_method == 'grads':
        #     G = kwargs['grads_matrix']
        #     P = G
        # elif self.subspace_constr_method == 'iterates_grads':
        #     G = kwargs['grads_matrix']
        #     X = kwargs['updates_matrix']
        #     P = np.hstack((G, X))
        # elif self.subspace_constr_method == 'iterates_grads_diagnewtons':
        #     raise NotImplementedError('Subspace construction not yet implemented in projected case.')
        #     G = kwargs['grads_matrix']
        #     X = kwargs['updates_matrix']
        #     D = kwargs['hess_diag_dirs_matrix']
        #     P = np.hstack((G, X, D))
        # elif self.subspace_constr_method == 'random':
        #     # In this case no problem information is used in building the subspace basis
        #     P = None
            
    # Initialise the vectors to be stored throughout the algorithm (if any)
    def init_stored_vectors(self, grad_vec):
        # At iteration 0 we do not have any update vectors yet
        X = None
        
        # Not implemented
        D = None

        if self.subspace_no_grads > 0:
            G = np.array(grad_vec, ndmin=2)
            G = G.reshape(-1, 1)
        else:
            G = None

        # if self.subspace_constr_method == 'grads':
        #     G = np.array(grad_vec, ndmin=2) # store gradient vectors
        #     G = G.reshape(-1, 1)
        #     X = None
        #     D = None
        # elif self.subspace_constr_method == 'iterates_grads':
        #     G = np.array(grad_vec, ndmin=2) # store gradient vectors
        #     G = G.reshape(-1, 1)
        #     X = np.array(x, ndmin=2) # store iterates
        #     X = X.reshape(-1, 1)
        #     D = None
        # elif self.subspace_constr_method == 'iterates_grads_diagnewtons':
        #     raise NotImplementedError('Not implemented!')
        #     G = np.array(grad_vec, ndmin=2) # store gradient vectors
        #     G = G.reshape(-1, 1)
        #     X = np.array(x, ndmin=2) # store iterates
        #     X = X.reshape(-1, 1)
        #     D = np.linalg.solve(np.diag(np.diag(hess_f_x)), grad_vec)
        #     D = D.reshape(-1, 1)
        # elif self.subspace_constr_method == 'random':
        #     G = None
        #     X = None
        #     D = None
        
        return G, X, D

    # Update stored vectors required for subspace constructions.
    def update_stored_vectors(self, x_update, proj_grad, full_grad_prev, Q_prev,
                              grads_matrix,
                              updates_matrix,
                              hess_diag_dirs_matrix):
        G = grads_matrix
        X = updates_matrix
        D = hess_diag_dirs_matrix

        if G is not None:
            if G.shape[1] == self.subspace_no_grads:
                G = np.delete(G, 0, 1) # delete first (oldest) column
            if (not self.random_proj) and self.reproject_grad:
                reproj_grad = Q_prev @ np.transpose(Q_prev) @ full_grad_prev
                G[:, -1] = reproj_grad # re-assign last projected gradient
            G = np.hstack((G, proj_grad.reshape(-1, 1))) # append newest

        if X is not None:
            if X.shape[1] == self.subspace_no_updates:
                X = np.delete(X, 0, 1) # delete first (oldest) column
            X = np.hstack((X, x_update.reshape(-1, 1))) # append newest
        else:
            if self.subspace_no_updates > 0: # Initial update
                X = np.array(x_update.reshape(-1, 1)).reshape(-1, 1)

        if D is not None:
            raise NotImplementedError('Not yet implemented.')

        # if self.subspace_constr_method == 'grads':
        #     if G.shape[1] == self.no_problem_dirs:
        #         G = np.delete(G, 0, 1) # delete first (oldest) column
        #     if (not self.random_proj) and self.reproject_grad:
        #         reproj_grad = Q_prev @ np.transpose(Q_prev) @ full_grad_prev
        #         G[:,-1] = reproj_grad # must re-assign the final column at this stage to be reprojected grad
        #     G = np.hstack((G, proj_grad.reshape(-1,1))) # append newest gradient
        # elif self.subspace_constr_method == 'iterates_grads':
        #     if 2 * G.shape[1] == self.no_problem_dirs:
        #         G = np.delete(G, 0, 1) # delete first (oldest) column
        #         X = np.delete(X, 0, 1)
        #     if (not self.random_proj) and self.reproject_grad:
        #         reproj_grad = Q_prev @ np.transpose(Q_prev) @ full_grad_prev
        #         G[:,-1] = reproj_grad # must re-assign the final column at this stage to be reprojected grad
        #     G = np.hstack((G, proj_grad.reshape(-1,1))) # append newest gradient
        #     X = np.hstack((X, x_update.reshape(-1,1))) # append newest iterate
        # elif self.subspace_constr_method == 'iterates_grads_diagnewtons':
        #     raise Exception('Implementation of "iterates_grads_diagnewtons" subspace construction in the projected case not yet thought out!')
        #     if 3 * G.shape[1] == self.no_problem_dirs:
        #         G = np.delete(G, 0, 1) # delete first (oldest) column
        #         X = np.delete(X, 0, 1)
        #         D = np.delete(D, 0, 1)
        #     G = np.hstack((G, proj_grad.reshape(-1,1))) # append newest gradient
        #     X = np.hstack((X, x_update.reshape(-1,1))) # append newest iterate
        #     D = np.hstack((D, np.linalg.solve(np.diag(np.diag(hess_f_x)), proj_grad).reshape(-1, 1))) # append newest crude diagonal Newton direction approximation
        
        # Return updated matrices
        return G, X, D

    def print_iter_info(self, terminal: bool, k: int, x, f_x, norm_full_grad, step_size):
        x_str = ", ".join([f"{xi:7.4f}" for xi in x])
        info_str = f"k = {k:4} || x = [{x_str}] || f(x) = {f_x:8.6e} || g_norm = {norm_full_grad:8.6e}"
        
        if not terminal:
            info_str += f" || t = {step_size:8.6f}"

        if terminal:
            print('------------------------------------------------------------------------------------------')
            print('TERMINATED')
            print('------------------------------------------------------------------------------------------')
            print(info_str)
            print('------------------------------------------------------------------------------------------')
            print()
            print()
        else:
            print(info_str)

    # Optimisation loop
    def optimise(self, x0):
        x = x0
        f_x = self.func(x)
        full_grad = self.grad_func(x)
        norm_full_grad = np.linalg.norm(full_grad)

        # The first projected gradient takes a projection from an entirely random matrix, always
        proj_grad = self.project_gradient(full_grad, random_proj=True)

        # IN FUTURE will want to implement Hessian actions product using Hessian actions (B-vector products), see autograd.hessian_vector_products
        if self.use_hess:
            hess_f_x = self.hess_func(x)
            full_B = hess_f_x
        else:
            raise NotImplementedError('Have not (yet!) implemented methods with user-provided B approximations to the Hessian!')

        # For later plotting
        f_vals_list = [f_x]
        update_norms_list = []
        direction_norms_list = []
        angles_to_full_grad_list = []
        full_grad_norms_list = [np.linalg.norm(full_grad)]
        proj_grad_norms_list = [np.linalg.norm(proj_grad)]

        G, X, D = self.init_stored_vectors(grad_vec=proj_grad)
        
        Q, last_cond_no, last_P_rank = self.update_subspace(grads_matrix=G,
                                                            updates_matrix=X,
                                                            hess_diag_dirs_matrix=D)

        cond_nos_list = [last_cond_no]
        P_ranks_list = [last_P_rank]

        # Project B matrix
        # Later may want to do this using Hessian actions in the case where Hessian information is used at all.
        proj_B = np.transpose(Q) @ (full_B @ Q)

        # Regularise the projected/reduced Hessian approximation if needed.
        proj_B = self.regularise_hessian(proj_B)

        for k in range(self.max_iter):
            # Termination
            if norm_full_grad < self.tol:
                if self.verbose:
                    self.print_iter_info(terminal=True, k=k, x=x, f_x=f_x,
                                         norm_full_grad=norm_full_grad,
                                         step_size=step_size)
                break
            
            full_grad_prev = full_grad

            # Compute upcoming update
            direction = - Q @ np.linalg.inv(proj_B) @ np.transpose(Q) @ full_grad
            step_size = self.backtrack_armijo(x, direction, f_x, full_grad)
            x_update = step_size * direction

            # Compute upcoming update's info:
            last_update_norm = np.linalg.norm(x_update)
            last_angle_to_full_grad = np.arccos(np.dot(direction, -full_grad) / (np.linalg.norm(direction) * np.linalg.norm(full_grad))) * 180 / np.pi
            
            # Print iteration info if applicable
            if self.verbose and k % self.iter_print_gap == 0:
                self.print_iter_info(terminal=False, k=k, x=x,
                                     f_x=f_x, norm_full_grad=norm_full_grad,
                                     step_size=step_size)
            
            # Update iterate
            x = x + x_update

            # Compute basic quantities at new iterate
            f_x = self.func(x)
            full_grad = self.grad_func(x)
            norm_full_grad = np.linalg.norm(full_grad)
            proj_grad = self.project_gradient(full_grad, random_proj=self.random_proj, Q_prev=Q)
            if self.use_hess:
                hess_f_x = self.hess_func(x)
                full_B = hess_f_x
            else:
                raise NotImplementedError('Quasi-Newton stuff not yet implemented!')

            # Update stored vectors required for subspace constructions.
            G, X, D = self.update_stored_vectors(x_update=x_update,
                                                 proj_grad=proj_grad,
                                                 full_grad_prev=full_grad_prev,
                                                 Q_prev=Q, grads_matrix=G,
                                                 updates_matrix=X,
                                                 hess_diag_dirs_matrix=D)
            
            # Update subspace basis matrix
            Q, last_cond_no, last_P_rank = self.update_subspace(grads_matrix=G,
                                                                updates_matrix=X,
                                                                hess_diag_dirs_matrix=D)

            # Append info for later plotting
            proj_B = np.transpose(Q) @ (full_B @ Q)
            proj_B = self.regularise_hessian(proj_B)
            direction_norms_list.append(np.linalg.norm(direction))
            update_norms_list.append(last_update_norm)
            angles_to_full_grad_list.append(last_angle_to_full_grad)
            f_vals_list.append(f_x)
            full_grad_norms_list.append(np.linalg.norm(full_grad))
            proj_grad_norms_list.append(np.linalg.norm(proj_grad))
            cond_nos_list.append(last_cond_no)
            P_ranks_list.append(last_P_rank)

        # Convert these to arrays for later plotting
        f_vals = np.array(f_vals_list)
        update_norms = np.array(update_norms_list)
        direction_norms = np.array(direction_norms_list)
        full_grad_norms = np.array(full_grad_norms_list)
        proj_grad_norms = np.array(proj_grad_norms_list)
        angles_to_full_grad = np.array(angles_to_full_grad_list)
        cond_nos = np.array(cond_nos_list) # condition numbers of P matrix at each iteration
        P_ranks = np.array(P_ranks_list)

        return SolverOutput(solver=self, final_x=x, final_k=k, f_vals=f_vals, 
                            update_norms=update_norms,
                            direction_norms=direction_norms,
                            full_grad_norms=full_grad_norms,
                            proj_grad_norms=proj_grad_norms,
                            angles_to_full_grad=angles_to_full_grad,
                            cond_nos=cond_nos,
                            P_ranks=P_ranks)
