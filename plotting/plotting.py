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

FIGSIZE_REF = (17, 8)
LINE_WIDTH = 1
MARKER_SIZE = 1

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
def plot_scalar_vs_iteration(solver_outputs, attr_names: list, log_plot: bool, alpha: float=1, use_markers: bool=False, marker_str: str='o', labels=None):
    plt.figure(figsize=FIGSIZE_REF)
    
    if labels is None:
        try:
            labels = [f"Subspace dim = {solver_outputs[i].solver.subspace_dim}" for i in range(len(solver_outputs))]
        except:
            labels = [f"Solver {i}" for i in range(len(solver_outputs))]

    for attr_name in attr_names:
        for solver_output, label in zip(solver_outputs, labels):
            values = getattr(solver_output, attr_name)
            
            plt.plot(values, linestyle='None' if use_markers else '-', alpha=alpha, linewidth=LINE_WIDTH, marker=marker_str if use_markers else 'None', markersize=MARKER_SIZE, label=f"{label} ({attr_name})")
    
    # Construct title
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


# The below is intended to plot exaclty two attributes together
def twin_plot_scalar_vs_iteration(solver_outputs, attr_names: list,
                                  log_plots: list, alpha: float=1,
                                  use_markers: bool=False, marker_str: str='o',
                                  labels=None):
    
    fig, ax1 = plt.subplots(figsize=FIGSIZE_REF)
    
    if labels is None:
        try:
            labels = [f"Subspace dim = {solver_outputs[i].solver.subspace_dim}" for i in range(len(solver_outputs))]
        except:
            labels = [f"Solver {i}" for i in range(len(solver_outputs))]


    for solver_output, label in zip(solver_outputs, labels):
        values = getattr(solver_output, attr_names[0])
        
        ax1.plot(values, linestyle='None' if use_markers else '-',
                 alpha=alpha,
                 linewidth=LINE_WIDTH,
                 color='b',
                 marker=marker_str if use_markers else 'None',
                 markersize=MARKER_SIZE,
                 label=f"{label} ({attr_names[0]})")
    
    # Construct title
    title_str = ''
    for attr_name in attr_names:
        title_str += attr_name + ' '
    title_str += 'vs Iteration'
    ax1.set_title(title_str)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    
    if log_plots[0]:
        ax1.set_yscale('log')
    if log_plots[1]:
        ax2.set_yscale('log')
    

    for solver_output, label in zip(solver_outputs, labels):
        values = getattr(solver_output, attr_names[1])
        
        ax2.plot(values, linestyle='None' if use_markers else '-', alpha=alpha,
                 linewidth=LINE_WIDTH,
                 color='red',
                 marker=marker_str if use_markers else 'None',
                 markersize=MARKER_SIZE,
                 label=f"{label} ({attr_names[1]})")
    
    ax2.set_ylabel('Value', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()  # To ensure there's no overlap
    fig.legend()
    ax1.grid(True, which="both", ls="-")
