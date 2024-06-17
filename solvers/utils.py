class SolverOutput():
    def __init__(self, solver, final_x, final_k, f_vals, grad_vals=None, grad_norms=None):
        self.final_x = final_x
        self.final_k = final_k
        self.f_vals = f_vals
        self.grad_vals = grad_vals
        self.grad_norms = grad_norms
        self.solver = solver

# It may be useful to implement classical (unlimited memory) BFGS update below.
def BFGS_update():
    pass