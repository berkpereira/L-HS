"""
This script is largely based on the work from https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022). There is another module in this repo, common_directions.py, reflecting the algorithms proposed in that paper closely. This module adapts that with some changes.

Namely, here we implement a version of that algorithm that never uses full gradient information.
We also seek to make this version of the algorithm more general, including variants that do/do not use Hessian information, etc.
The common thread to all of these methods is that THE FULL GRADIENT IS NEVER USED IN ITSELF IN THE ALGORITHM,
except when using edge case algorithm parameters (namely 'dimension' of S_k
being equal to the ambient dimension).

For discussion of these ideas, see relevant docs/*.md file.
"""
from dataclasses import dataclass, fields
from typing import Any
import warnings

import autograd.numpy as np
from autograd import grad, hessian
from solvers.utils import SolverOutput, scaled_gaussian, haar, append_dirs

@dataclass
class ProjectedCommonDirectionsConfig:
    """
    obj: Objective class instance.
    subspace_constr_method: Determines method for updating subspace.
    subspace_dim: Dimension of subproblem subspace.
    reg_lambda: Minimum allowable (POSITIVE) eigenvalue of projected Hessian.
    
    reproject_grad: NOTE: relevant only to DETERMINISTIC variant. If False, each
    gradient is projected only once, using the Q matrix from the previous iter.
    If True, then each gradient is also reprojected, at the next iteration,
    using its own iteration's Q matrix.

    append_rand_dirs: NOTE: Determines how many random directions to append
    to the (otherwise deterministic) subspace matrix at each iteration.
    Uses solvers.utils.append_dirs.
    """
    obj: Any                    
    reg_lambda: float
    
    subspace_no_grads: int = None
    subspace_no_updates: int = None
    subspace_no_random: int = None
    subspace_frac_grads: float = None
    subspace_frac_updates: float = None
    subspace_frac_random: float = None

    direction_str: str = 'sd' # options are {'newton', 'sd'}
    use_hess: bool = True
    random_proj: bool = True
    random_proj_dim: int = None
    random_proj_dim_frac: float = None
    reproject_grad: bool = False
    ensemble: str = ''
    inner_use_full_grad: bool = True

    # Determines whether to orthogonalise (QR) P_k, the subspace matrix.
    orth_P_k: bool = True
    # Determines whether P_k columns are ordinarily normalised to unit Euclidian norm.
    normalise_P_k_cols: bool = False
    
    # Constants --- CFS framework and others
    beta: float = 0.001
    tau: float = 0.5
    c_const: int = 0 # Positive integer. May also be set to np.inf to recover usual backtracking process
    alpha_max: float = 100 # Ceiling on step size parameter
    N_try: int = 1 # Number of allowable step retries for each subspace until success
    p_const: int = 1 # Positive integer, used in setting initial alpha

    
    # Passable attributes
    tol: float = 1e-6
    max_iter: int = 1000
    deriv_budget: int = None
    equiv_grad_budget: float = None
    iter_print_gap: int = 50
    verbose: bool = False

    def __post_init__(self):
        if not self.inner_use_full_grad:
            raise Exception('This feature (self.inner_use_full_grad == False) is no longer in use!')
        if not self.random_proj:
            raise Exception('This feature (self.random_proj == False) is no longer in use!')
        
        if ((self.subspace_frac_grads is not None and self.subspace_no_grads is not None) or
            (self.subspace_frac_updates is not None and self.subspace_no_updates is not None) or
            (self.subspace_frac_random is not None and self.subspace_no_random is not None) or
            (self.random_proj_dim_frac is not None and self.random_proj_dim is not None)):
            raise Exception('Cannot specify numbers of directions directly and as fractions of ambient dimension simultaneously!')

        # If fractions are specified, use them to set the integer attributes
        if self.subspace_frac_grads is not None:
            prospective_subspace_no_grads = self.subspace_frac_grads * self.obj.input_dim
            if int(prospective_subspace_no_grads) == prospective_subspace_no_grads:
                self.subspace_no_grads = int(prospective_subspace_no_grads)
            else:
                raise Exception(f"""Specified fraction of gradient directions does NOT give integer number of directions!
                                Gradients fraction: {self.subspace_frac_grads}. Ambient dimension: {self.obj.input_dim}""")
        if self.subspace_frac_updates is not None:
            prospective_subspace_no_updates = self.subspace_frac_updates * self.obj.input_dim
            if int(prospective_subspace_no_updates) == prospective_subspace_no_updates:
                self.subspace_no_updates = int(prospective_subspace_no_updates)
            else:
                raise Exception(f"""Specified fraction of update directions does NOT give integer number of directions!
                                Updates fraction: {self.subspace_frac_updates}. Ambient dimension: {self.obj.input_dim}""")
        if self.subspace_frac_random is not None:
            prospective_subspace_no_random = self.subspace_frac_random * self.obj.input_dim
            if int(prospective_subspace_no_random) == prospective_subspace_no_random:
                self.subspace_no_random = int(prospective_subspace_no_random)
            else:
                raise Exception(f"""Specified fraction of random directions does NOT give integer number of directions!
                                Random fraction: {self.subspace_frac_random}. Ambient dimension: {self.obj.input_dim}""")
        if self.random_proj_dim_frac is not None:
            prospective_random_proj_dim = self.random_proj_dim_frac * self.obj.input_dim
            if int(prospective_random_proj_dim) == prospective_random_proj_dim:
                self.random_proj_dim = int(prospective_random_proj_dim)
            else:
                raise Exception(f"""Specified fraction of random projection dimension does NOT give integer number of dimensions!
                                Random proj dim fraction: {self.random_proj_dim_frac}. Ambient dimension: {self.obj.input_dim}""")
        
        # If integer attributes are specified, set the fractional attributes accordingly
        if self.subspace_frac_grads is None:
            self.subspace_frac_grads = self.subspace_no_grads / self.obj.input_dim
        if self.subspace_frac_updates is None:
            self.subspace_frac_updates = self.subspace_no_updates / self.obj.input_dim
        if self.subspace_frac_random is None:
            self.subspace_frac_random = self.subspace_no_random / self.obj.input_dim
        if self.random_proj_dim_frac is None:
            self.random_proj_dim_frac = self.random_proj_dim / self.obj.input_dim

        # Overall dimension of subspace
        self.subspace_dim = self.subspace_no_grads + self.subspace_no_updates + self.subspace_no_random

        # Handle the relationship between deriv_budget and equiv_grad_budget
        if self.equiv_grad_budget is not None and self.deriv_budget is not None:
            raise Exception('Cannot specify derivative budget and equivalent gradient budget simultaneously!')
        if self.equiv_grad_budget is not None:
            self.deriv_budget = int(self.equiv_grad_budget * self.obj.input_dim)
        elif self.deriv_budget is not None:
            self.equiv_grad_budget = self.deriv_budget / self.obj.input_dim
        else:
            raise Exception("Either deriv_budget or equiv_grad_budget must be specified.")
        
        # More straightforward stuff, CFS constants
        self.nu = self.tau ** (-self.c_const) # 'Forwardtracking' factor

    def __str__(self):
        # These attributes should play no role for our purposes (consistent line plot colouring)
        passable_attrs = ['obj', 'verbose', 'deriv_budget', 'equiv_grad_budget',
                          'iter_print_gap', 'random_proj_dim', 'max_iter',
                          'tol', 'subspace_no_grads','subspace_no_updates',
                          'subspace_no_random']
        if self.subspace_no_grads == 0: # 'tilde projections' do not come up
            passable_attrs.extend(['random_proj_dim_frac', 'ensemble'])
        if self.direction_str != 'newton':
            passable_attrs.extend(['reg_lambda', 'use_hess']) # only comes in for Newton-like searches

        # If we are orthogonalising P_k, then the column-wise normalisation
        # setting is redundant.
        if self.orth_P_k:
            passable_attrs.append('normalise_P_k_cols')

        # NOTE: edge case --- Lee2022's CommonDirections
        if self.random_proj_dim == self.obj.input_dim: # recover Lee2022's CommonDirections
            passable_attrs.append('ensemble') # ensemble plays no role

        # NOTE: edge case --- full-space/classical linesearch method
        if self.subspace_dim == self.obj.input_dim:
            passable_attrs.extend(['random_proj_dim_frac', 'subspace_frac_grads',
                                   'subspace_frac_updates', 'subspace_frac_random',
                                   'random_proj', 'ensemble', 'inner_use_full_grad',
                                   'orth_P_k'])

        if self.subspace_frac_grads > 0: # projections 'make sense'
            if self.random_proj:
                passable_attrs.append('reproject_grad')
            else:
                passable_attrs.append('ensemble')
        else: # no tilde projections are ever computed
            passable_attrs.extend(['ensemble', 'reproject_grad', 'random_proj'])
        
        attributes = []
        for field in fields(self):
            name = field.name
            if name in passable_attrs:
                continue
            value = getattr(self, name)
            if isinstance(value, float):
                value = format(value, '.4g')  # Format float with a consistent representation
            elif isinstance(value, int):
                value = str(value)  # Convert int to string
            attributes.append(f"{name}={value}")
        return f"{self.__class__.__name__}({', '.join(attributes)})"

class ProjectedCommonDirections:
    def __init__(self, config: ProjectedCommonDirectionsConfig):
        # Set all attributes given in ProjectedCommonDirectionsConfig
        for key, value in config.__dict__.items():
            setattr(self, key, value)

        # Store the config object itself as an attribute
        self.config = config

        # Distinguish case when no problem information is used in P_k, i.e.,
        # where P_k is fully random.
        if self.subspace_no_grads == 0 and self.subspace_no_updates == 0:
            self.subspace_constr_method = 'random'
        else:
            self.subspace_constr_method = None

        if self.subspace_no_grads == 0 and self.subspace_constr_method != 'random':
            raise Exception('Not good to NOT have fully randomised P_k yet have no directions based on gradient information!')

        if self.subspace_constr_method == 'random' and (not self.inner_use_full_grad):
            raise Exception('If P_k fully randomised, must use full gradients in expression for search direction and backtracking!')
        
        # Number of P_k columns relying on problem/algorithm information
        # of some kind. 
        self.no_problem_dirs = self.subspace_no_grads + self.subspace_no_updates

        if self.no_problem_dirs <= 0 and self.subspace_constr_method != 'random':
            raise Exception('It makes no sense to have no problem information in subspace construction when not using a purely randomised subspace approach!')
        
        # Checks on gradient reprojection and whether subspace dimension allows it
        if self.reproject_grad and (not self.random_proj) and self.subspace_no_grads <= 1:
            raise Exception("Gradients can only be reprojected if we're storing more than one for each subspace!")
        
        # Set up relevant problem callables
        self.func = self.obj.func # callable objective func
        self.grad_func = self.obj.grad_func
        self.hess_func = self.obj.hess_func # for now have it as a full Hessian; later may use autograd.hessian_vector_product

        # Increment number of derivatives computed in each iteration depending on
        # algorithm variant in use
        if self.subspace_constr_method == 'random':
            self.deriv_per_iter = self.subspace_dim + 1
        else:
            if self.random_proj: # random tilde grad projection
                if self.inner_use_full_grad:
                    self.deriv_per_iter = (self.subspace_dim - self.subspace_no_grads) + self.random_proj_dim + 1
                else:
                    self.deriv_per_iter = self.random_proj_dim
            else: # deterministic tilde grad projection
                raise Exception('"Deterministic" projections no longer in use!')
                if self.reproject_grad:
                    if self.inner_use_full_grad:
                        self.deriv_per_iter = 2 * self.subspace_dim + 1
                    else:
                        self.deriv_per_iter = 2 * self.subspace_dim
                else:
                    if self.inner_use_full_grad:
                        self.deriv_per_iter = 2 * self.subspace_dim + 1
                    else:
                        self.deriv_per_iter = self.subspace_dim
        # NOTE: edge case of a full-space method
        if self.subspace_dim == self.obj.input_dim:
            self.deriv_per_iter = self.obj.input_dim
        
        if self.deriv_per_iter > self.obj.input_dim:
            raise Exception("Method uses as many or more derivatives per iteration than if using full space method!")
            warnings.warn("Using as many or more derivatives per iteration than if using full space method!", UserWarning)

    # Draw a TALL sketching matrix from random ensemble.
    def draw_sketch(self):
        if self.random_proj_dim == self.obj.input_dim: # Recover Lee2022's CommonDirections
            return np.identity(self.obj.input_dim) # ensemble should play no role here.
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
            # print('USING HESSIAN REGULARISATION!') # Notify
            B = B + (self.reg_lambda - lambda_min) * np.identity(self.subspace_dim)
            
        return B

    # This method takes in the current iterate and performs a backtracking
    # Armijo linesearch along the specified search direction until the Armijo
    # condition is satisfied (self attributes used where appropriate).
    def backtrack_armijo(self, x, direction, f_x, full_grad, proj_grad):
        alpha = self.alpha_init

        if self.inner_use_full_grad:
            grad_vec = full_grad
        else:
            grad_vec = proj_grad
        
        direction = np.squeeze(direction) # turn into vector
        while self.func(x + alpha * direction) > f_x + self.beta * alpha * np.dot(grad_vec, direction):
            alpha *= self.tau
        
        return alpha
    
    # This function simply checks for satisfaction of Armijo condition. Returns bool.
    def check_armijo_condition(self, x, alpha, direction, f_x, full_grad, proj_grad) -> bool:
        if self.inner_use_full_grad:
            grad_vec = full_grad
        else:
            grad_vec = proj_grad
        direction = np.squeeze(direction) # turn into vector
        check_bool = (self.func(x + alpha * direction) <= f_x + self.beta * alpha * np.dot(grad_vec, direction))
        return check_bool
        
    # The below method is new compared to the common_directions.py module.
    # It plays the role of determining how "projected gradients" are defined.
    # This refers to what I usually denote by $\tilde{\nabla}f(x_k)$.
    def project_gradient(self, full_grad, random_proj, **kwargs):
        if random_proj:
            # Test for edge case where we recover full gradient information:
            if self.random_proj_dim == self.obj.input_dim:
                W = np.eye(self.obj.input_dim)
            else:
                W = self.draw_sketch()
        else: # 'Deterministic' projection of gradients for subspace construction.
            # Test for edge case where we recover full gradient information:
            if self.subspace_dim == self.obj.input_dim:
                W = np.eye(self.obj.input_dim)
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
            # Add orthogonal random directions as necessary
            Q = append_dirs(curr_mat=None,
                                ambient_dim=self.obj.input_dim,
                                no_dirs=self.subspace_dim,
                                curr_is_orth=False,
                                orthogonalise=self.orth_P_k,
                                normalise_cols=self.normalise_P_k_cols)
            return Q
        else:
            G = kwargs['grads_matrix']
            X = kwargs['updates_matrix']
            D = kwargs['hess_diag_dirs_matrix']
            
            arrays = [arr for arr in [G, X, D] if arr is not None]
            P = np.hstack(arrays)
            
            # Add orthogonal random directions as necessary
            Q = append_dirs(curr_mat=P,
                                ambient_dim=self.obj.input_dim,
                                no_dirs=self.subspace_dim - P.shape[1],
                                curr_is_orth=False,
                                orthogonalise=self.orth_P_k,
                                normalise_cols=self.normalise_P_k_cols)
            return Q
            
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
        
        return G, X, D

    # Update stored vectors required for subspace constructions.
    def update_stored_vectors(self, x_update, proj_grad, full_grad_prev, Q_prev,
                              grads_matrix,
                              updates_matrix,
                              hess_diag_dirs_matrix,
                              reassigning_latest_proj_grad=False):
        """
        NOTE: reassigning_latest_proj_grad is a boolean meant to address
        situations where we are only reassigning the latest projected gradient
        because we have not managed a successful iteration with that P_k within
        N_try iterations!
        This situation is quite different from other instances of updating
        these vectors
        """
        
        G = grads_matrix
        X = updates_matrix
        D = hess_diag_dirs_matrix

        if not reassigning_latest_proj_grad:        
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
                else: # X is None and should stay that way
                    pass
        else: # in this case we simply reassign the final column of G, nothing else
            if G is not None:
                G[:, -1] = proj_grad

        if D is not None:
            raise NotImplementedError('Not yet implemented.')
        
        # Return updated matrices
        return G, X, D

    def search_dir(self, Q, full_grad, proj_grad, proj_B):
        if self.inner_use_full_grad:
            grad_vec = full_grad
        else:
            grad_vec = proj_grad
        
        if self.direction_str == 'newton':
            direction = - Q @ np.linalg.inv(proj_B) @ np.transpose(Q) @ grad_vec
        elif self.direction_str == 'sd':
            direction = - Q @ np.transpose(Q) @ grad_vec
            
        return direction

    def print_iter_info(self, last: bool, k: int, deriv_evals: int, x, f_x, norm_full_grad, step_size, terminated=None):
        x_str = ", ".join([f"{xi:7.4f}" for xi in x])
        info_str = f"k = {k:4} || deriv_evals = {deriv_evals:4.2e} || x = [{x_str}] || f(x) = {f_x:8.6e} || g_norm = {norm_full_grad:8.6e}"
        
        if not last:
            info_str += f" || t = {step_size:8.6f}"

        if last:
            if terminated:
                print('------------------------------------------------------------------------------------------')
                print('TERMINATED')
                print('------------------------------------------------------------------------------------------')
                print(info_str)
                print('------------------------------------------------------------------------------------------')
                print()
                print()
            else:
                print('------------------------------------------------------------------------------------------')
                print('ITER/DERIV BUDGET RUN OUT')
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
        alpha = self.alpha_max * (self.tau ** self.p_const)
        
        f_x = self.func(x)
        full_grad = self.grad_func(x)
        norm_full_grad = np.linalg.norm(full_grad)

        # The first projected gradient takes a projection from an entirely random matrix, always
        if self.subspace_no_grads > 0:
            proj_grad = self.project_gradient(full_grad, random_proj=True)
        else:
            proj_grad = None

        # IN FUTURE will want to implement Hessian actions product using Hessian actions (B-vector products), see autograd.hessian_vector_products
        if self.direction_str == 'newton':
            if self.use_hess:
                hess_f_x = self.hess_func(x)
                full_B = hess_f_x
            else:
                raise NotImplementedError('Have not (yet!) implemented methods with user-provided B approximations to the Hessian!')

        G, X, D = self.init_stored_vectors(grad_vec=proj_grad)
        
        Q = self.update_subspace(grads_matrix=G,
                                 updates_matrix=X,
                                 hess_diag_dirs_matrix=D)
        
        # Project B matrix
        # Later may want to do this using Hessian actions in the case where Hessian information is used at all.
        if self.direction_str == 'newton':
            proj_B = np.transpose(Q) @ full_B @ Q

            # Regularise the projected/reduced Hessian approximation if needed.
            proj_B = self.regularise_hessian(proj_B)
        else:
            proj_B = None

        # For later plotting
        f_vals_list = [f_x]
        update_norms_list = []
        direction_norms_list = []
        angles_to_full_grad_list = []
        full_grad_norms_list = [np.linalg.norm(full_grad)]
        if self.subspace_no_grads > 0:
            proj_grad_norms_list = [np.linalg.norm(proj_grad)]
        cond_nos_list = [last_cond_no]
        P_ranks_list = [last_P_rank]
        P_norms_list = [last_P_norm]

        # Initialise iteration count metrics
        k = 0
        deriv_eval_count = 0
        deriv_evals_list = [0]
        j_try = 0
        terminated = False


        store_all_full_grads = [full_grad]
        store_all_proj_grads = [proj_grad]
        store_all_x = [x]

        # Start loop
        # NOTE: Not even allowing deriv evals to EVER go over self.deriv_budget.
        while (k < self.max_iter) and (deriv_eval_count + self.deriv_per_iter < self.deriv_budget):
            # Termination
            if norm_full_grad < self.tol:
                terminated = True
                break
            
            full_grad_prev = full_grad

            # Compute upcoming update
            direction = self.search_dir(Q, full_grad, proj_grad, proj_B)
            x_update = direction * alpha
            armijo_satisfied = self.check_armijo_condition(x, alpha, direction,
                                                           f_x, full_grad,
                                                           proj_grad)
            
            # Edge case for data recording
            if k == 0:
                last_update_norm = np.linalg.norm(x_update)
                last_angle_to_full_grad = np.arccos(np.dot(direction, -full_grad) / (np.linalg.norm(direction) * np.linalg.norm(full_grad))) * 180 / np.pi

            j_try += 1
            if armijo_satisfied: # Successful iteration
                
                # Update iterate and step size parameter alpha
                x = x + x_update
                alpha = np.min((self.alpha_max, self.nu * alpha))
                j_try = 0
                store_all_x.append(x)

                # Compute upcoming update's info:
                last_update_norm = np.linalg.norm(x_update)
                last_angle_to_full_grad = np.arccos(np.dot(direction, -full_grad) / (np.linalg.norm(direction) * np.linalg.norm(full_grad))) * 180 / np.pi

                # Compute basic quantities at new iterate
                f_x = self.func(x)
                full_grad = self.grad_func(x)
                if self.subspace_no_grads > 0:
                    proj_grad = self.project_gradient(full_grad, random_proj=self.random_proj, Q_prev=Q)

                if self.direction_str == 'newton':
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
                Q = self.update_subspace(grads_matrix=G,
                                         updates_matrix=X,
                                         hess_diag_dirs_matrix=D)
                
                # Count added derivative/actions costs
                deriv_eval_count += self.deriv_per_iter
                
                # Update 2nd order matrix
                if self.direction_str == 'newton':
                    proj_B = np.transpose(Q) @ (full_B @ Q)
                    proj_B = self.regularise_hessian(proj_B)
            else: # Unsuccessful iteration
                # x is unchanged
                # Decrease step size parameter alpha (backtracking)
                alpha = self.tau * alpha
                if j_try == self.N_try:
                    # Must reproject the current gradient
                    if self.subspace_no_grads > 0:
                        proj_grad = self.project_gradient(full_grad, random_proj=self.random_proj, Q_prev=Q)
                    # Update just the last projected gradient!
                    G, X, D = self.update_stored_vectors(x_update=x_update,
                                                        proj_grad=proj_grad,
                                                        full_grad_prev=full_grad_prev,
                                                        Q_prev=Q, grads_matrix=G,
                                                        updates_matrix=None,
                                                        hess_diag_dirs_matrix=None,
                                                        reassigning_latest_proj_grad=True)
                    # Update subspace basis matrix
                    Q = self.update_subspace(grads_matrix=G,
                                             updates_matrix=X,
                                             hess_diag_dirs_matrix=D)
                    
                    # Update derivative evals count
                    deriv_eval_count += self.random_proj_dim + self.subspace_no_random

                    # Reset backtracking counter
                    j_try = 0
            


            # Print iteration info if applicable
            if self.verbose and k % self.iter_print_gap == 0:
                self.print_iter_info(last=False, k=k, x=x, deriv_evals=deriv_eval_count,
                                     f_x=f_x, norm_full_grad=norm_full_grad,
                                     step_size=alpha)

            # Append info for later plotting
            store_all_full_grads.append(full_grad)
            norm_full_grad = np.linalg.norm(full_grad)
            
            if self.subspace_no_grads > 0:
                store_all_proj_grads.append(proj_grad)

            direction_norms_list.append(np.linalg.norm(direction))
            update_norms_list.append(last_update_norm)
            angles_to_full_grad_list.append(last_angle_to_full_grad)
            f_vals_list.append(f_x)
            full_grad_norms_list.append(np.linalg.norm(full_grad))
            if proj_grad is not None:
                proj_grad_norms_list.append(np.linalg.norm(proj_grad))
            cond_nos_list.append(last_cond_no)
            P_ranks_list.append(last_P_rank)
            P_norms_list.append(last_P_norm)

            # Update iteration count metrics
            k += 1
            deriv_evals_list.append(deriv_eval_count)


        if self.verbose:
            self.print_iter_info(last=True, terminated=terminated, k=k,
                                 deriv_evals=deriv_eval_count,
                                 x=x, f_x=f_x, norm_full_grad=norm_full_grad,
                                 step_size=alpha)

        # Convert these to arrays for later plotting
        f_vals = np.array(f_vals_list)
        update_norms = np.array(update_norms_list)
        direction_norms = np.array(direction_norms_list)
        full_grad_norms = np.array(full_grad_norms_list)
        if proj_grad is not None:
            proj_grad_norms = np.array(proj_grad_norms_list)
        else:
            proj_grad_norms = full_grad_norms
        angles_to_full_grad = np.array(angles_to_full_grad_list)
        cond_nos = np.array(cond_nos_list) # condition numbers of P matrix at each iteration
        P_ranks = np.array(P_ranks_list)
        P_norms = np.array(P_norms_list)
        deriv_evals = np.array(deriv_evals_list)

        return SolverOutput(solver=self, final_f_val=f_vals[-1], final_x=x, final_k=k,
                            deriv_evals=deriv_evals,
                            f_vals=f_vals, 
                            update_norms=update_norms,
                            direction_norms=direction_norms,
                            full_grad_norms=full_grad_norms,
                            proj_grad_norms=proj_grad_norms,
                            angles_to_full_grad=angles_to_full_grad,
                            cond_nos=cond_nos,
                            P_ranks=P_ranks,
                            P_norms=P_norms)
