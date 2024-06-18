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
        

# It may be useful to implement classical (unlimited memory) BFGS B_k update below.
def BFGS_update():
    pass