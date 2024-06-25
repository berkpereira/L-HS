import scipy, scipy.optimize, scipy.linalg
import autograd.numpy as np

class SolverOutput():
    def __init__(self, solver, final_x, final_k, f_vals, update_norms=None, **kwargs):
        self.solver = solver
        self.final_x = final_x
        self.final_k = final_k
        self.f_vals = f_vals
        self.update_norms = update_norms # stores the norms of all the iterate update vectors

        # Any other data structures we may want to use
        for key, value in kwargs.items():
            setattr(self, key, value)
        

# Can use scipy's implementation of a strong-Wolfe-condition-ensuring
# linesearch (provided direction is a descent direction, of course).
# We use this in (L)BFGS linesearch methods to ensure positive-definiteness of
# hessian approximations.
def strong_wolfe_linesearch(func, grad_func, x, direction, c1, c2, max_iter):
    return scipy.optimize.line_search(func, grad_func, x,
                                        direction, c1=c1, c2=c2,
                                        maxiter=max_iter)


# TALL(!) scaled Gaussian sketching matrix!
# Subspace dimension taken to be the number of columns, n
def scaled_gaussian(m, n):
    return np.random.normal(scale=np.sqrt(1 / n), size=(m, n))

# NOTE: The function below is adapted from Mezzadri2007, Sec. 5
# (arXiv:math-ph/0609050)
# Generates a random orthonormal matrix with dimensions (m, n)
def haar(m, n):
    z = np.random.randn(m,m) / np.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)
    
    # Select first n rows to return
    return q[:, :n]

# The below is based on Algorithm 5 from https://doi.org/10.1007/s10107-022-01836-1
# It returns an orthonormal matrix including the directions of curr_mat's columns
# along with no_dirs (int) random directions.
def append_orth_dirs(curr_mat: np.ndarray, no_dirs: int, curr_is_orth: bool):
    n = curr_mat.shape[0] # column/ambient dimension
    A = np.random.randn(n, no_dirs)

    # curr_is_orth: bool. Input argument which specifies whether curr_mat is
    # an orthonormal matrix.
    if curr_is_orth:
        curr_mat_orth = curr_mat
    else:
        curr_mat_orth, _ = np.qr(curr_mat)
    
    # orthogonalise directions in A versus curr_mat
    A = A - curr_mat_orth @ np.transpose(curr_mat_orth) @ curr_mat
    
    new_dirs_mat, _ = np.qr(A)

    return np.hstack((curr_mat_orth, new_dirs_mat))
