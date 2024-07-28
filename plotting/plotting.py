import matplotlib.pyplot as plt
import autograd.numpy as np
from solvers.utils import SolverOutput
import hashlib
from solvers.utils import normalise_loss
from results.results_utils import get_hashed_filename

# Enable LaTeX rendering, etc.
plt.rcParams.update({
    'font.size': 11,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "axes.grid": True,
    'grid.alpha': 0.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# "Default" graphic parameters
FIGSIZE_REF = (17, 8.5)
LINE_WIDTH = 1
MARKER_SIZE = 1

# This function returns a hex number colour code from a hash (e.g., of a 
# solver's configuration's string representation)
def config_str_to_color(config_str: str):
    long_hash = get_hashed_filename(config_str)

    # Use the first 6 characters of the hash for the color
    color = f'#{long_hash[:6]}'
    return color

# Function to convert a config string to a linestyle
def config_str_to_linestyle(config_str: str):
    # List of available linestyles
    LINESTYLES = ['-', '--', '-.', ':']
    # Create a SHA-256 hash of the string representation of the configuration
    long_hash = get_hashed_filename(config_str)

    # Use the first character of the hash to determine the linestyle
    linestyle_index = int(long_hash[0], 16) % len(LINESTYLES)
    return LINESTYLES[linestyle_index]

# This function generates an appropriate label string given a SolverOutput object.
def solver_loss_label(solver_out: SolverOutput,
                      normalise_P_k_dirs_vs_dimension: bool,
                      normalise_S_k_dirs_vs_dimension: bool,
                      suppress_dir: bool=False,
                      suppress_sketch_size: bool=False,
                      suppress_c_const: bool=False,
                      suppress_N_try: bool=False):
    
    ambient_dim = solver_out.solver.obj.input_dim

    # NOTE: edge case --- full-space method
    if solver_out.solver.subspace_dim == ambient_dim:
        full_space_method = True
    else:
        full_space_method = False
    
    # NOTE: edge case --- Lee2022's CommonDirections
    if (solver_out.solver.random_proj and (solver_out.solver.random_proj_dim == ambient_dim)):
        lee_common_method = True
    else:
        lee_common_method = False
    
    # NOTE: edge case --- randomised subspace method
    if solver_out.solver.subspace_frac_grads == 0 and solver_out.solver.subspace_frac_updates == 0:
        random_subspace_method = True
    else:
        random_subspace_method = False

    if solver_out.solver.direction_str == 'newton':
        direction_str_formatted = solver_out.solver.direction_str.capitalize()
    elif solver_out.solver.direction_str == 'sd':
        direction_str_formatted = solver_out.solver.direction_str.upper()

    if full_space_method:
        new_label_template = """Full-space method
        Search: {direction_str_formatted}"""

        new_label = new_label_template.format(
            direction_str_formatted=direction_str_formatted
        )
        
        return new_label

    no_sub_grads   = solver_out.solver.subspace_no_grads
    no_sub_updates = solver_out.solver.subspace_no_updates
    no_sub_random  = solver_out.solver.subspace_no_random

    if normalise_P_k_dirs_vs_dimension:
        no_sub_grads      /= ambient_dim
        no_sub_updates    /= ambient_dim
        no_sub_random     /= ambient_dim
        no_sub_grads_str   = f'${no_sub_grads*100:.0f}\%$'
        no_sub_grads_eq    = '$=$' if no_sub_grads == np.round(no_sub_grads, 2) else r'$\approx$'
        no_sub_updates_str = f'${no_sub_updates*100:.0f}\%$'
        no_sub_updates_eq  = '$=$' if no_sub_updates == np.round(no_sub_updates, 2) else r'$\approx$'
        no_sub_random_str  = f'${no_sub_random*100:.0f}\%$'
        no_sub_random_eq   = '$=$' if no_sub_random == np.round(no_sub_random, 2) else r'$\approx$'
    else:
        no_sub_grads_str   = f'${no_sub_grads}$'
        no_sub_updates_str = f'${no_sub_updates}$'
        no_sub_random_str  = f'${no_sub_random}$'
        no_sub_grads_eq    = '$=$'
        no_sub_updates_eq  = '$=$'
        no_sub_random_eq   = '$=$'

    if normalise_S_k_dirs_vs_dimension:
        S_k_dim = solver_out.solver.random_proj_dim / ambient_dim
        S_k_dim_str = f'${S_k_dim*100:.0f}\%$'
        S_k_eq = '$=$'if S_k_dim == np.round(S_k_dim, 2) else r'$\approx$'
    else:
        S_k_dim = solver_out.solver.random_proj_dim
        S_k_dim_str = f'${S_k_dim}$'
        S_k_eq = '$=$'
    

    if lee_common_method:
        grad_dirs_str  = r'$\nabla{f}(x_k)$'
        label_lines = [r"""L-CommDir"""]
    elif random_subspace_method:
        grad_dirs_str  = None
        label_lines = [r"""Random subspace method"""]
    else:
        grad_dirs_str  = r'$\tilde{\nabla}f(x_k)$'
        label_lines = [r"""L-ProjCommDir"""]
    update_dirs_str = r'$s_k$'

    format_args = {}

    if solver_out.solver.subspace_frac_grads > 0:
        label_lines.append(r"""\# {grad_dirs_str} dirs.\ {no_sub_grads_eq} {no_sub_grads_str}""")
        format_args.update({'grad_dirs_str': grad_dirs_str,
                            'no_sub_grads_eq': no_sub_grads_eq,
                            'no_sub_grads_str': no_sub_grads_str})
    if solver_out.solver.subspace_frac_updates > 0:
        label_lines.append(r"""\# {update_dirs_str} dirs.\ {no_sub_updates_eq} {no_sub_updates_str}""")
        format_args.update({'update_dirs_str': update_dirs_str,
                            'no_sub_updates_eq': no_sub_updates_eq,
                            'no_sub_updates_str': no_sub_updates_str})
    if solver_out.solver.subspace_frac_random > 0:
        label_lines.append(r"""\# Random dirs.\ {no_sub_random_eq} {no_sub_random_str}""")
        format_args.update({'no_sub_random_eq': no_sub_random_eq,
                            'no_sub_random_str': no_sub_random_str})

    if not (suppress_sketch_size or (solver_out.solver.subspace_frac_grads == 0)):
        label_lines.append(r"""Sketch size {S_k_eq} {S_k_dim_str}""")
        format_args.update({'S_k_eq': S_k_eq,
                            'S_k_dim_str': S_k_dim_str})
    if not suppress_dir:
        label_lines.append(r"""Search: {direction_str_formatted}""")
        format_args.update({'direction_str_formatted': direction_str_formatted})
    
    if not suppress_c_const:
        if solver_out.solver.c_const == np.inf:
            c_const_str = r'$\infty$'
        else:
            c_const_str = f'${solver_out.solver.c_const}$'
        label_lines.append(r"""$c =$ {c_const_str}""")
        format_args.update({'c_const_str': c_const_str})

    if not suppress_N_try:
        if solver_out.solver.N_try == np.inf:
            N_try_str = r'$\infty$'
        else:
            N_try_str = f'${solver_out.solver.N_try}$'
        label_lines.append(r"""$\#_{N_try_subscript} =$ {N_try_str}""")
        format_args.update({'N_try_subscript': r'{\text{try}}', 'N_try_str': N_try_str})

    new_label_template = "\n".join(label_lines)

    new_label = new_label_template.format(**format_args)

    return new_label

# This function plots the loss vs iteration (or vs directional derivatives
# evaluated) for a number of solver output objects.
def plot_loss_vs_iteration(solver_outputs: list,
                           deriv_evals_axis: bool=False,
                           normalise_deriv_evals_vs_dimension: bool=False,
                           normalise_P_k_dirs_vs_dimension: bool=False,
                           normalise_S_k_dirs_vs_dimension: bool=False,
                           normalise_loss_data: bool=False,
                           labels=None):
    """
    Plot the loss (function values) vs iteration count for multiple solvers.

    Input arguments:
    solver_outputs: List of SolverOutput instances.
    deriv_evals_axis: Determines whether the horizontal axis is given in terms 
    directional derivative evaluations. If False, it is given in iterations instead.
    normalise_deriv_evals_vs_dimension: Determines whether, if deriv_evals_axis == True, the
    numbers of derivative evaluations are actually displayed as multiples of
    the problem ambient dimension.
    normalise_P_k_dirs_vs_dimension: Determines whether, in the plot labels, 
    the numbers of directions in P_k (from grads, steps, and random) are
    displayed as fractions of the ambient dimension, denoted by n.
    normalise_S_k_dirs_vs_dimension: Similar to normalise_P_k_dirs_vs_dimension,
    but referring to the 'dimension' of S_k instead
    labels: List of labels for each solver output.
    """
    plt.figure(figsize=FIGSIZE_REF)
    
    if labels is None:
        try: # If subspace dimension is a meaningful concept for the solver
            labels = []
            for solver_out in solver_outputs:
                new_label = solver_loss_label(solver_out,
                                              normalise_P_k_dirs_vs_dimension,
                                              normalise_S_k_dirs_vs_dimension)
                labels.append(new_label)
        except: # Generic chronological numbering
            labels = [f"Solver {i}" for i in range(len(solver_outputs))]

    # Track labelled solver configurations
    labelled_configs = set()

    for solver_output, label in zip(solver_outputs, labels):
        current_config_str = str(solver_output.solver.config)

        # Use the hash of the configuration to determine the color
        color = config_str_to_color(current_config_str)
        linestyle = config_str_to_linestyle(current_config_str)

        # Add label only if the configuration hasn't been labelled yet
        if current_config_str not in labelled_configs:
            labelled_configs.add(current_config_str)
            plot_label = label
        else:
            plot_label = None
        
        if normalise_loss_data:
            f0 = solver_output.f_vals[0]
            f_sol = solver_output.solver.obj.f_sol
            if f_sol is not None:
                y_data = normalise_loss(solver_output.f_vals, f_sol=f_sol, f0=f0)
            else:
                raise Exception('Trying to normalise loss but no best known solution was provided!')
        else:
            y_data = solver_output.f_vals

        if deriv_evals_axis:
            if normalise_deriv_evals_vs_dimension:
                ambient_dim = solver_output.solver.obj.input_dim
                equiv_grad_evals = solver_output.deriv_evals / ambient_dim
                plt.plot(equiv_grad_evals, y_data, linestyle=linestyle,
                         color=color, label=plot_label)
            else:
                plt.plot(solver_output.deriv_evals, y_data,
                         linestyle=linestyle, color=color, label=plot_label)
        else:
            plt.plot(y_data, linestyle=linestyle, color=color, label=plot_label)
    
    plt.yscale('log')
    if deriv_evals_axis:
        if normalise_deriv_evals_vs_dimension:
            plt.xlabel('Equivalent gradient evaluations')
            plt.title(f'Objective vs equivalent gradient evaluations. {solver_output.solver.obj.name}')
        else:
            plt.xlabel('(Directional) derivative evaluations')
            plt.title(f'Objective vs (directional) derivative evaluations. {solver_output.solver.obj.name}')
    else:
        plt.xlabel('Iteration')
        plt.title(f'Objective vs iteration. {solver_output.solver.obj.name}')
    if normalise_loss_data:
        plt.ylabel(r'$\bar{f}(x_k)$')
    else:
        plt.ylabel('$f(x_k)$')
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
        plt.title(f"{attr_name_avg} comparison")
        plt.ylabel(attr_name_avg)
        plt.yscale('log')
        plt.xlabel("Solvers")

# The below function is meant for plotting histograms of important scalar
# quantities associated with solver runs (as opposed to just their means)
def plot_run_histograms(raw_results: list, attr_names: list):
    """
    Plots histograms for the specified attributes of raw solver results.

    raw_results: List of two-tuples, one per solver, each containing Solver
    object and list of SolverOutput instances.
    attr_names: List of strings representing the attribute names to plot.
    """
    num_solvers = len(raw_results)
    num_attrs = len(attr_names)
    
    # Setting the figure size
    plt.figure(figsize=FIGSIZE_REF)

    for idx, attr_name in enumerate(attr_names):
        plt.subplot(num_attrs, 1, idx + 1)
        
        for solver_idx, (solver, solver_outputs) in enumerate(raw_results):
            color = config_str_to_color(str(solver.config))
            values = [getattr(output, attr_name) for output in solver_outputs]
            plt.hist(values, alpha=0.75, color=color,
                     label=f"Solver {solver_idx}: {solver.__class__.__name__}")
        
        plt.title(f"Histogram of {attr_name}")
        plt.xlabel(attr_name)
        plt.ylabel("Frequency")
        plt.legend()