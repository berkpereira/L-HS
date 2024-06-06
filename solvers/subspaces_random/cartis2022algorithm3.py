"""
Code based on Algorithm 3 from Cartis, Fowkes, and Shao, "Randomised subspace methods for non-convex optimization, with applications to nonlinear least-squares", 2022. https://arxiv.org/abs/2211.09873

In my own bibliography, this paper's shorthand designation is "cartis2022", hence the nomenclature in this file.

Throughout, the ROWS of S_k are used as the basis for the subspace used at iterate k. Lots of "transposes" are in order in translating between this and the CommonDirections algorithm coded elsewhere in this repository (itself based on https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022)).
"""
import autograd.numpy as np
from autograd import grad
from ..utils import SolverOutput

np.random.seed(42)

class Cartis2022Algorithm3:
    def __init__(self, obj, subspace_dim, gamma1, const_c, const_p, kappa_T, theta, alpha_max, alpha0, ensemble: str, hash_size=None, alpha=0.001, t_init=1, tau=0.5, tol=1e-6, max_iter=1000, iter_print_gap=20, verbose=False):
        pass
        self.obj = obj
        self.subspace_dim = subspace_dim
        self.const_c = const_c
        if not (isinstance(self.const_c, int) and self.const_c > 0):
            raise ValueError('const_c must be positive integer!')
        self.const_p = const_p
        if not (isinstance(self.const_p, int) and self.const_p > 0):
            raise ValueError('const_p must be positive integer!')
        self.gamma1 = gamma1
        if not (0 < self.gamma1 < 1):
            raise ValueError('gamma1 must be in (0,1).')
        self.gamma2 = 1 / (self.gamma1 ** const_c)
        self.theta = theta
        if not (0 < self.theta < 1):
            raise ValueError('theta must be in (0,1).')
        self.alpha_max = alpha_max
        if not self.alpha_max > 0:
            raise ValueError('alpha_max must be positive!')
        self.alpha0 = alpha0
        self.ensemble = ensemble
        # hashing ensembles not fully developed here yet
        if self.ensemble == 'hash':
            self.hash_size = hash_size
        self.func = self.obj.func # callable objective func
        self.alpha = alpha
        self.t_init = t_init
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.grad_func = grad(self.func)
        self.iter_print_gap = iter_print_gap
        self.verbose = verbose

    def draw_sketch(self) -> np.ndarray:
        if self.ensemble == 'scaled_gaussian':
            return np.random.normal(scale=np.sqrt(1 / self.subspace_dim), size=(self.obj.input_dim, self.subspace_dim))
        
    # This method will implement the local model function (\hat{m_k}) determined by the algorithm at each iterate.
    # This model is of reduced dimension relative to the ambient problem input dimension.
    # B stands for B_k, a PD approximation to the Hessian (think of quasi-Newton methods).
    def local_model(self, grad_hat: np.ndarray, B_hat: np.ndarray, s_hat: np.ndarray):
        return np.dot(grad_hat, s_hat) + 0.5 * np.dot(s_hat, B_hat @ s_hat)
    
    # This method the local REGularised model, denoted by \hat{q}_k in the reference paper
    def local_model_reg(self, grad_hat: np.ndarray, B_hat: np.ndarray, s_hat: np.ndarray, S: np.ndarray, alpha: float):
        return self.local_model(grad_hat, B_hat, s_hat) + (np.linalg.norm(np.transpose(S) @ s_hat) ** 2) / (2 * alpha)

    # This method will implement the (approximate) minimisation of the local model needed at each iterate
    # IN THIS ALGO, the local model is a convex quadratic, so it shouldn't be so so hard.
    def min_local_model(self):
        pass

    # This method will run the actual algorithm (outer iterations, if you will, while calling min_local_model to run the inner iterations) and return an approximate local minimiser of the function of interest
    def optimise(self):
        pass
