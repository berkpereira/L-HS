"""
Code based on Algorithm 3 from Cartis, Fowkes, and Shao, "Randomised subspace methods for non-convex optimization, with applications to nonlinear least-squares", 2022. https://arxiv.org/abs/2211.09873

In my own bibliography, this paper's shorthand designation is "cartis2022", hence the nomenclature in this file.

Throughout, the ROWS of S_k are used as the basis for the subspace used at iterate k. Lots of "transposes" are in order in translating between this and the CommonDirections algorithm coded elsewhere in this repository (itself based on https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022)).

At the moment, we make the reference paper's Algorithm 3 concrete by employing a backtracking Newton's method. This is because the local model at each iterate is a convex quadratic with essentially an already known/computed Hessian.

NEXT TO DO
NEXT TO DO
NEXT TO DO
NEXT TO DO
NEXT TO DO
NEXT TO DO
FORGO FULL GRADIENTS, IMPLEMENT AS DIRECTIONAL DERIVATIVES.
BUT::::PROFILER RESULTS SHOW THAT GRADIENTS TAKE UP BIG CHUNK OF TIME, AND ALSO THE NATURE OF OUR METHODS ARE TO AVOID EVER COMPUTING FULL GRADIENT!
"""

from dataclasses import dataclass

import autograd.numpy as np
from autograd import grad, hessian, make_jvp
from solvers.utils import SolverOutput
from solvers.linesearch_general import LinesearchGeneral
from problems.test_problems import Objective

np.random.seed(42)

@dataclass
class Cartis2022Algorithm3Config:
    obj: any
    subspace_dim: int
    gamma1: float
    const_c: int
    const_p: int
    kappa_T: float
    theta: float
    alpha_max: float
    ensemble: str
    hash_size: int = 0
    inner_beta: float = 0.001
    inner_t_init: float = 1
    inner_tau: float = 0.5
    tol: float = 1e-4
    outer_max_iter: int = 1_000
    inner_max_iter: int = 1_000
    iter_print_gap: int = 20
    verbose: bool = False

class Cartis2022Algorithm3:
    def __init__(self, config):
        for key, value in config.__dict__.items():
            setattr(self, key, value)

        if not (0 < self.gamma1 < 1):
            raise ValueError('gamma1 must be in (0,1).')
        self.gamma2 = 1 / (self.gamma1 ** self.const_c)

        if not (isinstance(self.const_c, int) and self.const_c > 0):
            raise ValueError('const_c must be positive integer!')

        if not (isinstance(self.const_p, int) and self.const_p > 0):
            raise ValueError('const_p must be positive integer!')
        if not (self.kappa_T > 0):
            raise ValueError('kappa_T must be positive!')
        self.kappa_T = self.kappa_T
        
        if not (0 < self.theta < 1):
            raise ValueError('theta must be in (0,1).')
        
        if not self.alpha_max > 0:
            raise ValueError('alpha_max must be positive!')
        
        self.alpha0 = self.alpha_max * (self.gamma1 ** self.const_p)
        
        self.func = self.obj.func # callable objective func

        self.grad_func = grad(self.func)
        

    # Below we specify methods necessary to perform the inner iterations' subproblem

    # Using Newton direction, as the Newton system is of low dimension and PD
    def inner_dir_func(self, obj, x, deriv_info: list, kwargs):
        gradient, hess = deriv_info[0], deriv_info[1]
        return - np.linalg.solve(hess, gradient)
    
    # We employ backtracking Armijo linesearch
    def inner_step_func(self, obj, s_hat, search_dir, deriv_info, kwargs): # "kwargs" reflects **kwargs from the LinesearchGeneral method
        tau = kwargs['tau']
        beta = kwargs['beta']
        
        t = self.inner_t_init
        trial_step = t * search_dir
        while obj.func(s_hat) - obj.func(s_hat + trial_step) < beta * np.dot(deriv_info[0], trial_step):
            t *= tau
        
        return t

    def inner_stop_crit_func(self, obj, max_iter, k, s_hat, m_s, deriv_info, kwargs):
        
        S = kwargs['S']

        # BELOW IS ORIGINAL, SUPPOSED ONE
        if (np.linalg.norm(deriv_info[0]) <= self.kappa_T * (np.linalg.norm(np.transpose(S) @ s_hat) ** 2) and obj.func(s_hat) <= obj.func(np.zeros(obj.input_dim))) or k >= self.inner_max_iter:
            return True
        else:
            return False


    # Draw a FAT sketching matrix from random ensemble.
    def draw_sketch(self) -> np.ndarray:
        # RECOVER FULL-DIMENSIONAL METHOD
        if self.subspace_dim == self.obj.input_dim:
            return np.identity(self.obj.input_dim)
        elif self.ensemble == 'scaled_gaussian':
            return np.random.normal(scale=np.sqrt(1 / self.subspace_dim), size=(self.subspace_dim, self.obj.input_dim))
        else:
            raise Exception('Unrecognised sketching matrix scheme!')

    # This method will implement the (approximate) minimisation of the local model needed at each iterate.
    # Notice that the local regularised model is a (convex) quadratic.
    def min_local_model(self, local_obj, g_vec: np.ndarray, hess: np.ndarray, S: np.ndarray, beta: float):
        # g_vec and hess are the relevant "parameters" of the regularised quadratic local model, see ref. paper, Algorithm 3
                
        # Below returns 1st- and 2nd-order derivative information of the regularised local quadratic model
        def deriv_info_func(s_hat):
            return [g_vec + (hess @ s_hat), hess]

        inner_optimiser = LinesearchGeneral(obj=local_obj, deriv_info_func=deriv_info_func, direction_func=self.inner_dir_func, step_func=self.inner_step_func, stop_crit_func=self.inner_stop_crit_func, max_iter=self.inner_max_iter, verbose=False, S=S, tau=self.inner_tau, beta=beta)

        inner_solver_output = inner_optimiser.optimise(np.zeros(self.subspace_dim))
        return inner_solver_output
    
    # The below method checks for sufficient decrease in the full-space objective following a lower-dimensional inner subproblem solution
    def check_suff_decrease(self, x, trial_step_hat, trial_step, local_model_func: callable):
        if self.obj.func(x) - self.obj.func(x + trial_step) >= self.theta * (local_model_func(np.zeros(self.subspace_dim)) - local_model_func(trial_step_hat)):
            return True
        else:
            return False

    # This method will run the actual algorithm (outer iterations, if you will, while calling min_local_model to run the inner iterations) and return an approximate local minimiser of the function of interest
    def optimise(self, x0):
        k = 0
        x = x0
        alpha = self.alpha_max
        f_x = self.func(x)
        f_vals = [f_x]
        
        # NOTICE HOW WE'RE ALLOWING OURSELVES TO COMPUTE THE FULL GRADIENT VECTOR HERE!
        # LATER ON WILL WANT TO REMOVE THESE, BUT FOR NOW IT'S EASIER TO GET THINGS WORKING THUSLY.
        grad_f_x = self.grad_func(x)

        for k in range(self.outer_max_iter):
            if np.linalg.norm(grad_f_x) < self.tol:
                if self.verbose:
                    x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                    print('------------------------------------------------------------------------------------------')
                    print('TERMINATED')
                    print('------------------------------------------------------------------------------------------')
                    print(f"Iteration {k:4}: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {np.linalg.norm(grad_f_x):10.6e}")
                break
            
            S = self.draw_sketch()
            
            # FOR NOW NOT WORRYING MUCH ABOUT B_k, set to zeros (still valid)
            B = np.zeros(shape=(self.obj.input_dim, self.obj.input_dim))

            red_reg_hessian = S @ (B + (1 / alpha) * np.identity(self.obj.input_dim)) @ np.transpose(S)

            red_grad = S @ grad_f_x
            
            # Regularised local quadratic model
            local_model_func = lambda s_hat: np.dot(red_grad, s_hat) + 0.5 * np.dot(s_hat, red_reg_hessian @ s_hat)
            local_model_obj = Objective(input_dim=self.subspace_dim, func=local_model_func)
            
            # Solve (approx.) inner regularised subproblem
            inner_solver_output = self.min_local_model(local_obj=local_model_obj, g_vec=red_grad, hess=red_reg_hessian, S=S, beta=self.inner_beta)

            # Store trial step
            trial_s_hat = inner_solver_output.final_x
            trial_s = np.transpose(S) @ trial_s_hat

            if self.check_suff_decrease(x=x, trial_step_hat=trial_s_hat, trial_step=trial_s, local_model_func=local_model_func): # successful iteration
                if self.verbose and k % self.iter_print_gap == 0:
                    x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                    # print(f"Iteration {k:4} SUCCESSFUL: x = [{x_str}], f(x) = {f_x:10.6e}, grad norm = {np.linalg.norm(grad_f_x):10.6e}, step size = {np.linalg.norm(trial_s):8.6f}, alpha = {alpha:5.3e}")
                    print(f"Iteration {k:4}   SUCCESSFUL: f(x) = {f_x:10.6e}, grad norm = {np.linalg.norm(grad_f_x):10.6e}, step size = {np.linalg.norm(trial_s):8.6f}, alpha = {alpha:5.3e}")
                x = x + trial_s
                
                f_x = self.func(x)
                f_vals.append(f_x)
                grad_f_x = self.grad_func(x)

                alpha = np.min((self.alpha_max, self.gamma2 * alpha))
            else: # unsuccessful iteration
                if self.verbose and k % self.iter_print_gap == 0:
                    x_str = ", ".join([f"{xi:8.4f}" for xi in x])
                    # print(f"Iteration {k:4} UNSUCCESSFUL: x = [{x_str}], f(x) = {f_x:10.6e}, alpha = {alpha:5.3e}")
                    print(f"Iteration {k:4} UNSUCCESSFUL: f(x) = {f_x:10.6e}, grad norm = {np.linalg.norm(grad_f_x):10.6e}, step size = {np.linalg.norm(trial_s):8.6f}, alpha = {alpha:5.3e}")
                # x goes without update
                alpha *= self.gamma1

                f_vals.append(f_x)
        
        return SolverOutput(solver=self, final_x=x, final_k=k, f_vals=f_vals)
            