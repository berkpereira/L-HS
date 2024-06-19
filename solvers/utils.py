import scipy.optimize

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