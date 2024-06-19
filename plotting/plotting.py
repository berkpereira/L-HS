import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams.update({
    'font.size': 11,
    "text.usetex": True,
    "font.family": "serif",
    "axes.grid": True,
    'grid.alpha': 0.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

FIGSIZE_REF = (12, 6)



def plot_loss_vs_iteration(solver_outputs, labels=None):
    """
    Plot the loss (function values) vs iteration count for multiple solvers.

    :param solver_outputs: List of SolverOutput instances.
    :param labels: List of labels for each solver output.
    """
    plt.figure(figsize=FIGSIZE_REF)
    
    if labels is None:
        try: # If subspace dimension is a meaningful concept for the solver
            labels = [f"Subspace dim = {solver_outputs[i].solver.subspace_dim}" for i in range(len(solver_outputs))]
        except: # Generic chronological numbering
            labels = [f"Solver {i}" for i in range(len(solver_outputs))]

    for solver_output, label in zip(solver_outputs, labels):
        plt.plot(solver_output.f_vals, linestyle='-', label=label)
    
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.title('Loss vs Iteration')
    plt.legend()
    plt.grid(True, which="both", ls="-")

# Below for plotting various quantities
def plot_scalar_vs_iteration(solver_outputs, attr_names: list, log_plot: bool, labels=None):
    plt.figure(figsize=FIGSIZE_REF)
    
    if labels is None:
        try:
            labels = [f"Subspace dim = {solver_outputs[i].solver.subspace_dim}" for i in range(len(solver_outputs))]
        except:
            labels = [f"Solver {i}" for i in range(len(solver_outputs))]

    for attr_name in attr_names:
        for solver_output, label in zip(solver_outputs, labels):
            values = getattr(solver_output, attr_name)
            plt.plot(values, linestyle='-', label=f"{label} ({attr_name})")
    

    title_str = ''
    for attr_name in attr_names:
        title_str += attr_name + ' '
        title_str += 'vs Iteration'
    
    plt.title(title_str)
    
    if log_plot:
        plt.yscale('log')
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, which="both", ls="-")