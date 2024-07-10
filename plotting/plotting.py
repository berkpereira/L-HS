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

# This function plots the loss vs iteration (or vs directional derivatives
# evaluated) for a number of solver output objects.
def plot_loss_vs_iteration(solver_outputs: list,
                           deriv_evals_axis: bool=False,
                           labels=None):
    """
    Plot the loss (function values) vs iteration count for multiple solvers.

    Input arguments:
    solver_outputs: List of SolverOutput instances.
    deriv_evals_axis: Determines whether the horizontal axis is given in terms 
    directional derivative evaluations. If False, it is given in iterations instead.
    labels: List of labels for each solver output.
    """
    plt.figure(figsize=FIGSIZE_REF)
    
    if labels is None:
        try: # If subspace dimension is a meaningful concept for the solver
            labels = [
                f"""\# sub grads = {solver_outputs[i].solver.subspace_no_grads},
                \# sub updates = {solver_outputs[i].solver.subspace_no_updates},
                \# sub random = {solver_outputs[i].solver.subspace_no_random}"""
                for i in range(len(solver_outputs))
                ]
        except: # Generic chronological numbering
            labels = [f"Solver {i}" for i in range(len(solver_outputs))]

    for solver_output, label in zip(solver_outputs, labels):
        if deriv_evals_axis:
            plt.plot(solver_output.deriv_evals, solver_output.f_vals, linestyle='-', label=label)
        else:
            plt.plot(solver_output.f_vals, linestyle='-', label=label)
    
    plt.yscale('log')
    if deriv_evals_axis:
        plt.xlabel('(Directional) derivatives evaluated')
        plt.title('Loss vs (directional) derivatives evaluated')
    else:
        plt.xlabel('Iteration')
        plt.title('Loss vs iteration')
    plt.ylabel('Function value')
    plt.legend()
    plt.grid(True, which="both", ls="-")

# This function plots arbitrary scalar array attributes of solver output objects.
def plot_scalar_vs_iteration(solver_outputs: list,
                             attr_names: list,
                             log_plot: bool,
                             alpha: float=1,
                             use_markers: bool=False,
                             marker_str: str='o', labels=None):
    """
    Input arguments:
    solver_outputs: list of SolverOutput objects.
    attr_names: list of strings determining attribute names to be retrived for plotting.
    log_plot: determines whether y axis is log scale.
    alpha: determines opacity of plots.
    use_markers: self-explanatory.
    marker_str: determines marker shape if markers in use.
    """

    plt.figure(figsize=FIGSIZE_REF)
    
    if labels is None:
        try:
            labels = [f"Subspace dim = {solver_outputs[i].solver.subspace_dim}" for i in range(len(solver_outputs))]
        except:
            labels = [f"Solver {i}" for i in range(len(solver_outputs))]

    for attr_name in attr_names:
        for solver_output, label in zip(solver_outputs, labels):
            values = getattr(solver_output, attr_name)
            
            if values is not None:
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

# The below is intended to plot exaclty two attributes in the same figure
def twin_plot_scalar_vs_iteration(solver_outputs: list, attr_names: list,
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
        
        if values is not None:
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
        try: # if nothing plotted, this will throw exception
            ax1.set_yscale('log')
        except:
            pass
    if log_plots[1]:
        try:
            ax2.set_yscale('log')
        except:
            pass
    

    for solver_output, label in zip(solver_outputs, labels):
        values = getattr(solver_output, attr_names[1])
        
        if values is not None:
            ax2.plot(values, linestyle='None' if use_markers else '-', alpha=alpha,
                    linewidth=LINE_WIDTH,
                    color='red',
                    marker=marker_str if use_markers else 'None',
                    markersize=MARKER_SIZE,
                    label=f"{label} ({attr_names[1]})")
    
    ax2.set_ylabel('Value', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    try:
        fig.tight_layout()  # To ensure there's no overlap
    except:
        pass
    fig.legend()
    ax1.grid(True, which="both", ls="-")

# This function is designed to use the output of the
# function solvers.utils.average_solver_runs
def plot_solver_averages(avg_results: dict, attr_names: list):
    """
    Plots bar charts for the specified attributes of SolverOutputAverage instances.
    
    Input arguments:
    avg_results: Dictionary as output by solvers.utils.average_solver_runs
    attr_names: List of strings representing the attribute names to plot
    """
    # Below retrieve a list of tuples [(Solver, SolverOutputAverage), ..., (Solver, SolverOutputAverage)]
    avg_results_list = avg_results['avg_results']

    num_solvers = len(avg_results_list)
    num_attrs = len(attr_names)
    
    # Setting the figure size
    plt.figure(figsize=FIGSIZE_REF)

    for idx, attr_name in enumerate(attr_names):
        # Creating a subplot for each attribute
        plt.subplot(num_attrs, 1, idx + 1)
        attr_name_avg = f'{attr_name}_avg'
        
        # Extracting attribute values for each solver
        attr_values = [getattr(solver_avg, attr_name_avg) for solver, solver_avg in avg_results_list]
        solver_labels = [f"Solver {i}" for i, (solver, solver_avg) in enumerate(avg_results_list)]
        
        # Plotting the bar chart
        plt.bar(solver_labels, attr_values, alpha=0.75)
        plt.title(f"{attr_name_avg.replace('_', ' ').capitalize()} comparison")
        plt.ylabel(attr_name_avg.replace('_', ' ').capitalize())
        plt.xlabel("Solvers")
        plt.grid(True)
    
    plt.tight_layout()
