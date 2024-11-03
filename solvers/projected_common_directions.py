"""
The common thread to all of these methods is that THE FULL GRADIENT IS NEVER USED IN ITSELF IN THE ALGORITHM,
except when using edge case algorithm parameters (namely 'dimension' of S_k
being equal to the ambient dimension).
"""
from dataclasses import dataclass, fields
import time
from typing import Any
import warnings

import autograd.numpy as np
from autograd import grad, hessian
from solvers.utils import SolverOutput, scaled_gaussian, haar, append_dirs, lsr1

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
    
    subspace_no_grads: int   = None
    subspace_no_updates: int = None
    subspace_no_random: int  = None

    subspace_no_grads_given_as_frac: bool   = None
    subspace_no_updates_given_as_frac: bool = None
    subspace_no_random_given_as_frac: bool  = None

    subspace_frac_grads: float   = None
    subspace_frac_updates: float = None
    subspace_frac_random: float  = None

    # NOTE: the below only in use if direction_str == 'newton' and use_hess is False.
    no_secant_pairs: int = 0

    direction_str: str = 'sd' # options are {'newton', 'sd'}
    use_hess: bool = True # if direction_str == 'newton' but this is False, the method uses a quasi-Newton approach
    random_proj_dim: int = None
    random_proj_dim_frac: float = None
    reproject_grad: bool = False
    ensemble: str = '' # NOTE: in {'haar', 'scaled_gaussian'}

    reg_lambda: float = 0

    # Determines whether to orthogonalise (QR) P_k, the subspace matrix.
    orth_P_k: bool = True
    # Determines whether P_k columns are ordinarily normalised to unit Euclidian norm.
    normalise_P_k_cols: bool = False
    
    # Constants --- CFS framework and others
    beta: float = 0.001
    tau: float = 0.5       # Float in (0, 1)
    c_const: int = np.inf  # POSITIVE integer. May also be set to np.inf to recover usual backtracking process
    N_try: int = 200         # Number of allowable step retries for each subspace until success
    alpha_max: float = 100 # Ceiling on step size parameter
    p_const: int = 1       # POSITIVE integer, used in setting initial alpha

    # 'Passable' attributes
    tol: float = 1e-6
    max_iter: int = 1000
    deriv_budget: int = None
    equiv_grad_budget: float = None
    iter_print_gap: int = 50
    verbose: bool = False
    timeout_secs: int = np.inf

    def __post_init__(self):
        if not (self.orth_P_k or self.normalise_P_k_cols):
            raise ValueError('Must have either self.orth_P_k or self.normalise_P_k_cols active!!!')
        if self.orth_P_k and self.normalise_P_k_cols:
            raise ValueError('It is redundant to have both orthogonalisation and explicit column normalisation!')

        if self.p_const < 1 or self.c_const < 1:
            raise ValueError('p and c constants must be POSITIVE integers!')
        
        if ((self.subspace_frac_grads is not None and self.subspace_no_grads is not None) or
            (self.subspace_frac_updates is not None and self.subspace_no_updates is not None) or
            (self.subspace_frac_random is not None and self.subspace_no_random is not None) or
            (self.random_proj_dim_frac is not None and self.random_proj_dim is not None)):
            raise Exception('Cannot specify numbers of directions directly and as fractions of ambient dimension simultaneously!')
        
        # NOTE: Determine whether the number of subspace directions of each kind
        # was user-specified as a number or as a fraction ('frac') of the
        # problem ambient dimension.
        if self.subspace_no_grads is None:
            self.subspace_no_grads_given_as_frac = True
        else:
            self.subspace_no_grads_given_as_frac = False
        if self.subspace_no_updates is None:
            self.subspace_no_updates_given_as_frac = True
        else:
            self.subspace_no_updates_given_as_frac = False
        if self.subspace_no_random is None:
            self.subspace_no_random_given_as_frac = True
        else:
            self.subspace_no_random_given_as_frac = False

        # Shorthand for quasi-Newton methods
        if self.direction_str == 'newton' and (not self.use_hess):
            self.quasi_newton = True
        else:
            self.quasi_newton = False

        # If fractions are specified, use them to set the integer attributes
        # NOTE: This only goes for cases where a solver is then to be run.
        # Otherwise --- when obj is None --- is when we are using config objects
        # in comparison of the same specification across different
        # problems (e.g. in plotting data profiles). In this case it would make
        # no sense to assign numbers of directions here.
        if self.obj is not None:
            if self.subspace_frac_grads is not None:
                self.subspace_no_grads = int(np.ceil(self.subspace_frac_grads * self.obj.input_dim))
            if self.subspace_frac_updates is not None:
                self.subspace_no_updates = int(np.ceil(self.subspace_frac_updates * self.obj.input_dim))
            if self.subspace_frac_random is not None:
                self.subspace_no_random = int(np.ceil(self.subspace_frac_random * self.obj.input_dim))
            if self.random_proj_dim_frac is not None:
                self.random_proj_dim = int(np.ceil(self.random_proj_dim_frac * self.obj.input_dim))
        
        # If integer attributes are specified, set the fractional attributes accordingly
        # NOTE: as in the above, we may have (obj is None) e.g. when plotting
        # data profiles, where the specification of numbers of dimensions across
        # different problems of generally different ambient dimensions makes
        # no sense.
        if self.obj is not None:
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

            if self.subspace_dim > self.obj.input_dim:
                raise Exception('Sum of subspace dimensions is larger than the problem ambient dimension!')

        # Handle the relationship between deriv_budget and equiv_grad_budget
        # NOTE: this is another aspect which we have no need for when
        # e.g. generating data profiles, i.e. when (self.obj is None)
        if self.obj is not None:
            if self.equiv_grad_budget is not None and self.deriv_budget is not None:
                raise Exception('Cannot specify derivative budget and equivalent gradient budget simultaneously!')
            if self.equiv_grad_budget is not None:
                self.deriv_budget = int(self.equiv_grad_budget * self.obj.input_dim)
            elif self.deriv_budget is not None:
                self.equiv_grad_budget = self.deriv_budget / self.obj.input_dim
            else:
                raise Exception("Either deriv_budget or equiv_grad_budget must be specified.")
        
        # NOTE: this may raise an exception in certain scenarios when creating a
        # config object from a string, where none of these specified (due to an
        # edge case where the method is a classical full-space one,
        # for instance)
        try: 
            if self.subspace_frac_grads + self.subspace_frac_updates + self.subspace_frac_random > 1:
                raise Exception('Total subspace fraction exceeds 1, this makes no sense!')
        except:
            pass
        
        # More straightforward stuff, CFS constants
        self.nu = self.tau ** (-self.c_const) # 'Forward-tracking' factor

    def __str__(self):
        # These attributes should play no role for our purposes (consistent line plot colouring)
        passable_attrs = ['obj', 'verbose', 'deriv_budget', 'equiv_grad_budget',
                          'iter_print_gap', 'random_proj_dim', 'max_iter',
                          'tol', 'timeout_secs',
                          'subspace_no_grads_given_as_frac',
                          'subspace_no_updates_given_as_frac',
                          'subspace_no_random_given_as_frac',
                          'quasi_newton']
        
        if self.subspace_no_grads_given_as_frac:
            passable_attrs.append('subspace_no_grads')
        else:
            passable_attrs.append('subspace_frac_grads')
        if self.subspace_no_updates_given_as_frac:
            passable_attrs.append('subspace_no_updates')
        else:
            passable_attrs.append('subspace_frac_updates')
        if self.subspace_no_random_given_as_frac:
            passable_attrs.append('subspace_no_random')
        else:
            passable_attrs.append('subspace_frac_random')

        if self.subspace_no_grads == 0: # 'tilde projections' do not come up
            passable_attrs.extend(['random_proj_dim_frac', 'ensemble'])
        if self.direction_str != 'newton':
            passable_attrs.extend(['reg_lambda', 'use_hess', 'no_secant_pairs']) # only comes in for Newton-like searches
        else: # in Newton-like methods...
            if self.use_hess: # NOT using quasi-Newton method
                passable_attrs.append('no_secant_pairs')

        # If we are orthogonalising P_k, then the column-wise normalisation
        # setting is redundant.
        if self.orth_P_k:
            passable_attrs.append('normalise_P_k_cols')

        # NOTE: edge case --- Lee2022's CommonDirections
        if self.random_proj_dim_frac == 1: # recover Lee2022's CommonDirections
            passable_attrs.append('ensemble') # ensemble plays no role

        # NOTE: edge case --- full-space/classical linesearch method.
        full_space = False
        try:
            if ((self.subspace_frac_grads is None and self.subspace_frac_updates is None and self.subspace_frac_random is None)):
                full_space = True
            
            # NOTE: this condition may raise exception if not all of these were
            # specified as fractions of the ambient dimension!
            elif (self.subspace_frac_grads + self.subspace_frac_updates + self.subspace_frac_random) == 1:
                full_space = True
        except:
            full_space = False

        if full_space:
            passable_attrs.extend(['random_proj_dim_frac', 'subspace_frac_grads',
                                'subspace_frac_updates', 'subspace_frac_random',
                                'ensemble', 'orth_P_k', 'reproject_grad'])
        else: # usual, not-full-space-method cases
            uses_projections = False
            try:
                if self.subspace_frac_grads > 0:
                    uses_projections = True
            except:
                if self.subspace_no_grads > 0:
                    uses_projections = True
            if uses_projections: # projections 'make sense'
                passable_attrs.append('reproject_grad')
            else: # no "tilde projections" are ever computed
                passable_attrs.extend(['ensemble', 'reproject_grad', 'random_proj_dim_frac'])
        
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

        if self.subspace_constr_method == 'random':
            raise Exception('If P_k fully randomised, must use full gradients in expression for search direction and backtracking!')
        
        # Number of P_k columns relying on problem/algorithm information
        # of some kind. 
        self.no_problem_dirs = self.subspace_no_grads + self.subspace_no_updates

        if self.no_problem_dirs <= 0 and self.subspace_constr_method != 'random':
            raise Exception('It makes no sense to have no problem information in subspace construction when not using a purely randomised subspace approach!')
        
        # Checks on gradient reprojection and whether subspace dimension allows it
        if self.reproject_grad and self.subspace_no_grads <= 1:
            raise Exception("Gradients can only be reprojected if we're storing more than one for each subspace!")
        
        # Set up relevant problem callables
        self.func = self.obj.func # callable objective func
        self.grad_func = self.obj.grad_func
        self.hess_func = self.obj.hess_func # for now have it as a full Hessian; later may use autograd.hessian_vector_product

        # Limited-memory Quasi-Newton initial Hessian approximation
        self.B0 = np.eye(self.obj.input_dim) # NOTE: keeping it simple

        # Increment number of derivatives computed in each iteration depending on
        # algorithm variant in use
        if self.subspace_constr_method == 'random':
            if self.direction_str == 'sd' or self.quasi_newton:
                self.deriv_per_succ_iter = self.subspace_dim
                self.deriv_per_unsucc_iter = self.subspace_dim
            elif self.direction_str == 'newton' and (not self.quasi_newton):
                self.deriv_per_succ_iter = self.subspace_dim + (self.subspace_dim + 1) * self.obj.input_dim
                self.deriv_per_unsucc_iter = (self.obj.input_dim + 1) * self.subspace_dim
        else:
            if self.direction_str == 'sd' or self.quasi_newton:
                if self.subspace_dim == self.obj.input_dim or self.random_proj_dim_frac == 1: # edge case, full space method
                    self.deriv_per_succ_iter = self.obj.input_dim
                    self.deriv_per_unsucc_iter = 0
                else: # usual cases
                    self.deriv_per_succ_iter = self.subspace_dim + self.random_proj_dim - 1
                    self.deriv_per_unsucc_iter = self.random_proj_dim + self.subspace_no_random
            elif self.direction_str == 'newton' and (not self.quasi_newton):
                if self.subspace_dim == self.obj.input_dim: # edge case, full space method
                    self.deriv_per_succ_iter = (self.obj.input_dim + 1) * self.obj.input_dim
                elif self.random_proj_dim_frac == 1:
                    self.deriv_per_succ_iter = self.obj.input_dim + (self.subspace_dim) * self.obj.input_dim
                    self.deriv_per_unsucc_iter = 0 + (self.subspace_no_random) * self.obj.input_dim
                else: # usual cases
                    self.deriv_per_succ_iter = self.subspace_dim + self.random_proj_dim + (self.subspace_dim + 1) * self.obj.input_dim
                    self.deriv_per_unsucc_iter = self.random_proj_dim + self.subspace_no_random + (self.subspace_no_random + 1) * self.obj.input_dim

        # NOTE: edge case of a full-space method
        if self.subspace_dim == self.obj.input_dim:
            self.deriv_per_succ_iter = self.obj.input_dim
        
        if ((self.direction_str == 'sd' and self.deriv_per_succ_iter > self.obj.input_dim) or
            (self.direction_str == 'newton' and self.deriv_per_succ_iter > (self.obj.input_dim + 1) * self.obj.input_dim)):
            raise Exception("Method uses more derivatives per iteration than if you used a full space method!")

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
            # print('USING REGULARISATION!') # Notify
            B = B + (self.reg_lambda - lambda_min) * np.identity(self.subspace_dim)
            
        return B

    # This method takes in the current iterate and performs a backtracking
    # Armijo linesearch along the specified search direction until the Armijo
    # condition is satisfied (self attributes used where appropriate).
    def backtrack_armijo(self, x, direction, f_x, full_grad, proj_grad):
        alpha = self.alpha_init
        
        direction = np.squeeze(direction) # turn into vector
        while self.func(x + alpha * direction) > f_x + self.beta * alpha * np.dot(full_grad, direction):
            alpha *= self.tau
        
        return alpha
    
    # This function simply checks for satisfaction of Armijo condition. Returns bool.
    def check_armijo_condition(self, x, alpha, direction, f_x, full_grad, proj_grad) -> bool:
        direction = np.squeeze(direction) # turn into vector
        check_bool = (self.func(x + alpha * direction) <= f_x + self.beta * alpha * np.dot(full_grad, direction))
        return check_bool
        
    # The below method is new compared to the common_directions.py module.
    # It plays the role of determining how "projected gradients" are defined.
    # This refers to what I usually denote by $\tilde{\nabla}f(x_k)$.
    def project_gradient(self, full_grad):
        # Test for edge case where we recover full gradient information:
        if self.random_proj_dim == self.obj.input_dim:
            W = np.eye(self.obj.input_dim)
        else:
            W = self.draw_sketch()

        proj_grad = W @ np.transpose(W) @ full_grad
        return proj_grad

    # Which basis of subspace to use in the method?
    def update_subspace(self, **kwargs):
        # method: str. Options:
        # method == 'grads', retain m past gradient vectors
        # method == 'iterates_grads', (34) from Lee et al., 2022
        # method == 'iterates_grads_diagnewtons', (35) from Lee et al., 2022

        # G_subspace should store some past PROJECTED gradient vectors.
        # X_subspace should store some past iterate update vectors.
        # D_subspace should store some past crude Newton direction
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
            G_subspace = kwargs['grads_matrix']
            X_subspace = kwargs['updates_matrix']
            D_subspace = kwargs['hess_diag_dirs_matrix']
            
            arrays = [arr for arr in [G_subspace, X_subspace, D_subspace] if arr is not None]
            P = np.hstack(arrays)
            
            # Add orthogonal random directions as necessary
            Q = append_dirs(curr_mat=P,
                                ambient_dim=self.obj.input_dim,
                                no_dirs=self.subspace_dim - P.shape[1], # NOTE how we fill in extra random directions if not enough problem history has been accumulated yet!
                                curr_is_orth=False,
                                orthogonalise=self.orth_P_k,
                                normalise_cols=self.normalise_P_k_cols)
            return Q
            
    # Initialise the vectors to be stored throughout the algorithm (if any)
    def init_stored_vectors_subspace(self, grad_vec):
        # At iteration 0 we do not have any update vectors yet
        X_subspace = None
        
        # Not implemented
        D_subspace = None

        if self.subspace_no_grads > 0:
            G_subspace = np.array(grad_vec, ndmin=2)
            G_subspace = G_subspace.reshape(-1, 1)
        else:
            G_subspace = None
        
        return G_subspace, X_subspace, D_subspace
    
    def init_stored_vectors_lsr1(self):
        # At iteration 0 we have none of the elements of a secant pair!
        # NOTE that this differs from the situation in the vectors stored for
        # subspace construction purposes, where the first projected gradient
        # is there right away.
        Y_lsr1 = None
        X_lsr1 = None
        return Y_lsr1, X_lsr1


    # Update stored vectors required for subspace constructions.
    def update_stored_vectors_subspace(self, new_X_vec, new_G_vec, full_grad_prev, Q_prev,
                              G, X, D,
                              no_grads_stored: int,
                              no_updates_stored: int,
                              reassigning_latest_proj_grad=False):
        """
        NOTE: reassigning_latest_proj_grad is a boolean meant to address
        situations where we are only reassigning the latest projected gradient
        because we have not managed a successful iteration with that P_k within
        N_try iterations!
        This situation is quite different from other instances of updating
        these vectors
        """


        if not reassigning_latest_proj_grad:        
            if G is not None:
                if G.shape[1] == no_grads_stored:
                    G = np.delete(G, 0, 1) # delete first (oldest) column

                G = np.hstack((G, new_G_vec.reshape(-1, 1))) # append newest
            
            # NOTE: now the other case; in secant pair matrices, we have
            # the initial 'gradient differences' matrix equal to None, 
            # similarly to how the X matrix is ALWAYS None in the
            # first iteration.
            else:
                if no_grads_stored > 0: # Initial secant pair
                    G = np.array(new_G_vec.reshape(-1, 1)).reshape(-1, 1)
                else: # G is None and should simply stay that way
                    pass

            if X is not None:
                if X.shape[1] == no_updates_stored:
                    X = np.delete(X, 0, 1) # delete first (oldest) column
                X = np.hstack((X, new_X_vec.reshape(-1, 1))) # append newest
            else:
                if no_updates_stored > 0: # Initial update
                    X = np.array(new_X_vec.reshape(-1, 1)).reshape(-1, 1)
                else: # X is None and should simply stay that way
                    pass
        else: # in this case we simply reassign the final column of G, nothing else
            if G is not None:
                G[:, -1] = new_G_vec

        if D is not None:
            raise NotImplementedError('Not yet implemented.')
        
        # Return updated matrices
        return G, X, D

    def update_stored_vectors_lsr1(self, new_X_vec, new_Y_vec, Y, X,
                                   current_B, following_success: bool):
        # NOTE: need a threshold for cut-off on the cosine of the relevant
        # angle. See Nocedal & Wright, 1st ed. §8.2, where 1e-8 is suggested
        lsr1_cos_min = 1e-8
        try:
            temp_vec = new_Y_vec - np.dot(current_B, new_X_vec)
            abs_cos = np.abs(np.dot(new_X_vec, temp_vec)) / (np.linalg.norm(new_X_vec) * np.linalg.norm(temp_vec))
            if abs_cos >= lsr1_cos_min:
                lsr1_update = True
            else:
                lsr1_update = False
        except:
            lsr1_update = False
        
        if not lsr1_update: # NOT taking this secant pair onboard; skip!
            return Y, X

        if Y is not None:
            if Y.shape[1] == self.no_secant_pairs:
                if following_success:
                    Y = np.delete(Y, 0, 1) # delete first (oldest) column
                else:
                    Y = np.delete(Y, Y.shape[1] - 1, 1) # delete last (most RECENT) column
            Y = np.hstack((Y, new_Y_vec.reshape(-1, 1))) # append to right-hand end
        
        # NOTE: now the other case; in secant pair matrices, we have
        # the initial 'gradient differences' matrix equal to None, 
        # similarly to how the X matrix is ALWAYS None in the
        # first iteration.
        else:
            if self.no_secant_pairs > 0: # Initial secant pair
                Y = np.array(new_Y_vec.reshape(-1, 1)).reshape(-1, 1)
            else: # Y is None and should simply stay that way
                pass

        # NOTE: there is only ever a need to update the steps matrix following
        # a successful iteration!
        if following_success:
            if X is not None:
                if X.shape[1] == self.no_secant_pairs:
                    X = np.delete(X, 0, 1) # delete first (oldest) column
                X = np.hstack((X, new_X_vec.reshape(-1, 1))) # append newest
            else:
                if self.no_secant_pairs > 0: # Initial update
                    X = np.array(new_X_vec.reshape(-1, 1)).reshape(-1, 1)
                else: # X is None and should simply stay that way
                    pass
        
        print(f'Columns in Y: {Y.shape[1]}. Columns in X: {X.shape[1]}.')

        return Y, X

    def search_dir(self, Q, full_grad, proj_grad, proj_B):
        
        if self.direction_str == 'newton':
            hat_direction = np.linalg.lstsq(proj_B, - np.transpose(Q) @ full_grad)[0]
            direction = np.dot(Q, hat_direction)
        elif self.direction_str == 'sd':
            direction = - Q @ np.transpose(Q) @ full_grad
            
        return direction

    def print_iter_info(self, last: bool, k: int, deriv_evals: int, x, f_x, norm_full_grad, step_size, terminated=None, timed_out=False):
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
            elif timed_out:
                print('------------------------------------------------------------------------------------------')
                print(f'TIMED OUT: {self.timeout_secs} SECONDS ELAPSED')
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
        
        # Set timer
        start_time = time.time()

        f_x = self.func(x)
        full_grad = self.grad_func(x)
        norm_full_grad = np.linalg.norm(full_grad)

        # The first projected gradient takes a projection from an entirely random matrix, always
        if self.subspace_no_grads > 0:
            proj_grad = self.project_gradient(full_grad)
            old_proj_grad = proj_grad
        else:
            proj_grad = None

        # Initialise vectors stored for subspace construction purposes
        G_subspace, X_subspace, D_subspace = self.init_stored_vectors_subspace(grad_vec=proj_grad)
        
        # Initialise vectors stored for quasi-Newton approximation purposes,
        # if applicable
        if self.quasi_newton:
            Y_lsr1, X_lsr1 = self.init_stored_vectors_lsr1()

        # IN FUTURE may want to implement Hessian actions product using Hessian actions (B-vector products), see autograd.hessian_vector_products
        if self.direction_str == 'newton':
            if self.use_hess:
                full_B = self.hess_func(x)
            else: # self.quasi_newton is True
                full_B = lsr1(B0=self.B0, Y=Y_lsr1, X=X_lsr1)
        
        Q = self.update_subspace(grads_matrix=G_subspace,
                                 updates_matrix=X_subspace,
                                 hess_diag_dirs_matrix=D_subspace)
        
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

        # Initialise iteration count metrics
        k = 0
        deriv_eval_count = 0
        deriv_evals_list = [0]
        j_try = 0
        terminated = False
        timed_out = False

        # Start loop
        # NOTE: Not even allowing deriv evals to EVER go over self.deriv_budget.
        while (k < self.max_iter) and (deriv_eval_count + self.deriv_per_succ_iter < self.deriv_budget):
            spent_derivatives = False
            # Termination
            if norm_full_grad < self.tol:
                terminated = True
                break
            if time.time() - start_time > self.timeout_secs:
                timed_out = True
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
            if armijo_satisfied: # NOTE: SUCCESSFUL ITERATION
                spent_derivatives = True
                # Update iterate and step size parameter alpha
                x = x + x_update
                alpha = np.min((self.alpha_max, self.nu * alpha))
                j_try = 0

                # Compute upcoming update's info:
                last_update_norm = np.linalg.norm(x_update)
                last_angle_to_full_grad = np.arccos(np.dot(direction, -full_grad) / (np.linalg.norm(direction) * np.linalg.norm(full_grad))) * 180 / np.pi

                # Compute basic quantities at new iterate
                f_x = self.func(x)
                full_grad = self.grad_func(x)
                if self.subspace_no_grads > 0:
                    old_proj_grad = proj_grad # The one which is left behind at a distinct iterate is the old projected gradient
                    proj_grad = self.project_gradient(full_grad)
                    grad_diff = proj_grad - old_proj_grad

                if self.direction_str == 'newton':
                    if self.use_hess:
                        full_B = self.hess_func(x)
                    else: # self.quasi_newton is True
                        full_B = lsr1(B0=self.B0, Y=Y_lsr1, X=X_lsr1)

                # Update stored vectors required for subspace constructions.
                G_subspace, X_subspace, D_subspace = self.update_stored_vectors_subspace(new_X_vec=x_update,
                                                    new_G_vec=proj_grad,
                                                    full_grad_prev=full_grad_prev,
                                                    Q_prev=Q,
                                                    G=G_subspace,
                                                    X=X_subspace,
                                                    D=D_subspace,
                                                    no_grads_stored=self.subspace_no_grads,
                                                    no_updates_stored=self.subspace_no_updates)
                
                if self.quasi_newton:
                    Y_lsr1, X_lsr1 = self.update_stored_vectors_lsr1(new_X_vec=x_update,
                                                                    new_Y_vec=grad_diff,
                                                                    Y=Y_lsr1,
                                                                    X=X_lsr1,
                                                                    current_B=full_B,
                                                                    following_success=True)

                # Update subspace basis matrix
                Q = self.update_subspace(grads_matrix=G_subspace,
                                         updates_matrix=X_subspace,
                                         hess_diag_dirs_matrix=D_subspace)
                
                # Count added derivative/actions costs
                deriv_eval_count += self.deriv_per_succ_iter
                
                # Update 2nd order matrix
                if self.direction_str == 'newton':
                    proj_B = np.transpose(Q) @ (full_B @ Q)
                    proj_B = self.regularise_hessian(proj_B)
            else: # NOTE: UNSUCCESSFUL ITERATION
                # x is unchanged
                # Decrease step size parameter alpha (backtracking)
                alpha = self.tau * alpha
                if j_try == self.N_try:
                    spent_derivatives = True
                    # Must reproject the current gradient
                    if self.subspace_no_grads > 0:
                        proj_grad = self.project_gradient(full_grad)
                        grad_diff = proj_grad - old_proj_grad # NOTE that old_proj_grad is a projected gradient from the last distinct iterate!
                    # Update just the last projected gradient!
                    G_subspace, X_subspace, D_subspace = self.update_stored_vectors_subspace(new_X_vec=x_update,
                                                        new_G_vec=proj_grad,
                                                        full_grad_prev=full_grad_prev,
                                                        Q_prev=Q,
                                                        G=G_subspace,
                                                        X=None,
                                                        D=None,
                                                        reassigning_latest_proj_grad=True,
                                                        no_grads_stored=self.subspace_no_grads,
                                                        no_updates_stored=self.subspace_no_updates)
                    
                    if self.quasi_newton:
                        Y_lsr1, X_lsr1 = self.update_stored_vectors_lsr1(new_X_vec=None,
                                                                        new_Y_vec=grad_diff,
                                                                        current_B=full_B,
                                                                        following_success=True)
                    # Update subspace basis matrix
                    Q = self.update_subspace(grads_matrix=G_subspace,
                                             updates_matrix=X_subspace,
                                             hess_diag_dirs_matrix=D_subspace)
                    
                    # Update derivative evals count
                    deriv_eval_count += self.deriv_per_unsucc_iter

                    # Reset backtracking counter
                    j_try = 0

            # Print iteration info if applicable
            if self.verbose and k % self.iter_print_gap == 0:
                self.print_iter_info(last=False, k=k, x=x, deriv_evals=deriv_eval_count,
                                     f_x=f_x, norm_full_grad=norm_full_grad,
                                     step_size=alpha)

            # Only keeping records if meaningful developments have happened.
            if spent_derivatives:
                direction_norms_list.append(np.linalg.norm(direction))
                update_norms_list.append(last_update_norm)
                angles_to_full_grad_list.append(last_angle_to_full_grad)
                f_vals_list.append(f_x)
                full_grad_norms_list.append(np.linalg.norm(full_grad))
                if proj_grad is not None:
                    proj_grad_norms_list.append(np.linalg.norm(proj_grad))
                deriv_evals_list.append(deriv_eval_count)
            
            norm_full_grad = np.linalg.norm(full_grad)
            # Update iteration count regardless
            k += 1

        if self.verbose:
            self.print_iter_info(last=True, terminated=terminated, timed_out=timed_out, k=k,
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
        deriv_evals = np.array(deriv_evals_list)

        return SolverOutput(solver=self, final_f_val=f_vals[-1], final_x=x, final_k=k,
                            deriv_evals=deriv_evals,
                            f_vals=f_vals, 
                            update_norms=update_norms,
                            direction_norms=direction_norms,
                            full_grad_norms=full_grad_norms,
                            proj_grad_norms=proj_grad_norms,
                            angles_to_full_grad=angles_to_full_grad)
