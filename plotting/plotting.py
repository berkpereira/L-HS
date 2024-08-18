import matplotlib.pyplot as plt
import autograd.numpy as np
import results.results_utils
from solvers.utils import SolverOutput
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import hashlib
from solvers.utils import normalise_loss
from results.results_utils import get_hashed_filename

# Enable LaTeX rendering, etc.
plt.rcParams.update({
    'font.size': 9,
    "text.usetex": True,
    "text.latex.preamble": r"""\usepackage{amsmath}""",
    "font.family": "serif",
    "axes.grid": True,
    'grid.alpha': 0.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# "Default" graphic parameters
FIGSIZE_REF = (5.9, 2.9)
FIGSIZE_HALF = (FIGSIZE_REF[0] / 2, FIGSIZE_REF[1])
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
def solver_label(config: ProjectedCommonDirectionsConfig,
                 include_Pk_orth: bool=False,
                 include_sketch_size: bool=False,
                 include_c_const: bool=False,
                 include_N_try: bool=False,
                 include_ensemble: bool=False):

    # NOTE: edge case --- full-space method
    if config.subspace_frac_grads == 1 or config.subspace_frac_updates == 1 or config.subspace_frac_random == 1 or (config.subspace_frac_grads is None and config.subspace_frac_updates is None and config.subspace_frac_random is None):
        full_space_method = True
    else:
        full_space_method = False
    
    # NOTE: edge case --- Lee2022's CommonDirections
    if (config.random_proj and (config.random_proj_dim_frac == 1)):
        lee_common_method = True
    else:
        lee_common_method = False

    # NOTE NOTE NOTE: probably best to forgo this special case distinction, for
    # clarity/making it easier for the reader
    lee_common_method = False
    
    # NOTE: edge case --- randomised subspace method
    if config.subspace_frac_grads == 0 and config.subspace_frac_updates == 0:
        random_subspace_method = True
    else:
        random_subspace_method = False

    if full_space_method:
        if config.direction_str == 'newton':
            direction_str_formatted = 'Regularised Newton'
        elif config.direction_str == 'sd':
            direction_str_formatted = 'SD'
        new_label_template = """{direction_str_formatted} method"""

        new_label = new_label_template.format(
            direction_str_formatted=direction_str_formatted
        )
        
        return new_label

    frac_sub_grads   = config.subspace_frac_grads
    frac_sub_updates = config.subspace_frac_updates
    frac_sub_random  = config.subspace_frac_random


    # frac_sub_grads_str   = f'${frac_sub_grads*100:.0f}\%$'
    # frac_sub_grads_eq    = '$=$' if frac_sub_grads == np.round(frac_sub_grads, 2) else r'$\approx$'
    # frac_sub_updates_str = f'${frac_sub_updates*100:.0f}\%$'
    # frac_sub_updates_eq  = '$=$' if frac_sub_updates == np.round(frac_sub_updates, 2) else r'$\approx$'
    # frac_sub_random_str  = f'${frac_sub_random*100:.0f}\%$'
    # frac_sub_random_eq   = '$=$' if frac_sub_random == np.round(frac_sub_random, 2) else r'$\approx$'

    S_k_dim_frac = config.random_proj_dim_frac
    S_k_dim_str = f'$m_s = {S_k_dim_frac*100:.0f}\%$'
    S_k_eq = '$=$'if S_k_dim_frac == np.round(S_k_dim_frac, 2) else r'$\approx$'

    format_args = {}

    if lee_common_method:
        if config.direction_str == 'sd':
            label_lines = [r"""\verb|L-CommDir-SD-{grad_frac_str}.{update_frac_str}.{random_frac_str}|"""]
        elif config.direction_str == 'newton':
            label_lines = [r"""\verb|L-CommDir-N-{grad_frac_str}.{update_frac_str}.{random_frac_str}|"""]
    elif random_subspace_method:
        grad_dirs_str  = None
        if config.direction_str == 'sd':
            label_lines = [r"""RS-SD"""]
        elif config.direction_str == 'newton':
            label_lines = [r"""RS-N"""]
    else:
        if config.direction_str == 'sd':
            label_lines = [r"""\verb|L-HS-SD-{grad_frac_str}.{update_frac_str}.{random_frac_str}|"""]
        elif config.direction_str == 'newton':
            label_lines = [r"""\verb|L-HS-N-{grad_frac_str}.{update_frac_str}.{random_frac_str}|"""]
    
    if not random_subspace_method:
        format_args.update({'grad_frac_str': str(int(config.subspace_frac_grads * 100)),
                            'update_frac_str': str(int(config.subspace_frac_updates * 100)),
                            'random_frac_str': str(int(config.subspace_frac_random * 100))})
        
    if include_Pk_orth and (not full_space_method):
        if config.orth_P_k:
            label_lines.append(r"""$P_k$ orthonormal""")
        elif config.normalise_P_k_cols:
            label_lines.append(r"""$P_k$ \textbf{{not}} orthonormal""")

    if include_ensemble and (config.random_proj_dim_frac not in (0, 1)):
        if config.ensemble == 'scaled_gaussian':
            label_lines.append(r"""$S_k$ Gaussian""")
        elif config.ensemble == 'haar':
            label_lines.append(r"""$S_k$ orthonormal""")
        else:
            raise Exception('Unrecognised ensemble!')

    if (include_sketch_size and config.subspace_frac_grads != 0 and (not lee_common_method)):
        # label_lines.append(r"""Sketch size {S_k_eq} {S_k_dim_str}""")
        # format_args.update({'S_k_eq': S_k_eq,
        #                     'S_k_dim_str': S_k_dim_str})
        label_lines.append(r"""{S_k_dim_str}""")
        format_args.update({'S_k_dim_str': S_k_dim_str})
    
    if include_c_const:
        if config.c_const == np.inf:
            c_const_str = r'$\infty$'
        else:
            c_const_str = f'${config.c_const}$'
        label_lines.append(r"""$c =$ {c_const_str}""")
        format_args.update({'c_const_str': c_const_str})

    if include_N_try:
        if config.N_try == np.inf:
            N_try_str = r'$\infty$'
        else:
            N_try_str = f'${config.N_try}$'
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
                           normalise_loss_data: bool=False,
                           labels=None,
                           include_Pk_orth: bool=False,
                           include_sketch_size: bool=False,
                           include_c_const: bool=False,
                           include_N_try: bool=False,
                           include_ensemble: bool=False,
                           figsize='full_width',
                           label_ncol: int=1):
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
    if figsize == 'full_width':
        fig = plt.figure(figsize=FIGSIZE_REF)
    elif figsize == 'half_width':
        fig = plt.figure(figsize=FIGSIZE_HALF)
    else:
        fig = plt.figure(figsize=figsize)
    
    if labels is None:
        try:
            labels = []
            for solver_out in solver_outputs:
                new_label = solver_label(solver_out.solver.config,
                                         include_Pk_orth=include_Pk_orth,
                                         include_sketch_size=include_sketch_size,
                                         include_c_const=include_c_const,
                                         include_N_try=include_N_try,
                                         include_ensemble=include_ensemble)
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
                plt.step(equiv_grad_evals, y_data, linestyle=linestyle,
                         color=color, label=plot_label, where='post')
            else:
                plt.step(solver_output.deriv_evals, y_data,
                         linestyle=linestyle, color=color, label=plot_label, where='post')
        else:
            plt.step(y_data, linestyle=linestyle, color=color, label=plot_label, where='post')
    
    plt.yscale('log')
    if deriv_evals_axis:
        if normalise_deriv_evals_vs_dimension:
            plt.xlabel('Equivalent gradient evaluations')
        else:
            plt.xlabel('(Directional) derivative evaluations')
            if normalise_loss_data:
                plt.title(f'Normalised objective vs (directional) derivative evaluations. {solver_output.solver.obj.name}')
            else:
                plt.title(f'Objective vs (directional) derivative evaluations. {solver_output.solver.obj.name}')
    else:
        plt.xlabel('Iteration')
        # plt.title(f'Objective vs iteration. {solver_output.solver.obj.name}')
    if normalise_loss_data:
        plt.ylabel(r'Normalised objective')
    else:
        plt.ylabel('Objective')
    plt.legend(ncol=label_ncol)
    plt.grid(True, which="both", ls="-")
    return fig

def plot_data_profiles(success_dict: dict,
                       include_Pk_orth: bool=False,
                       include_sketch_size: bool=False,
                       include_c_const: bool=False,
                       include_N_try: bool=False,
                       include_ensemble: bool=False,
                       figsize='full_width',
                       label_ncol: int=1):
    """
    Plot data profiles from the success_dict generated by generate_data_profiles.
    
    success_dict: Dictionary containing success rates for different solvers and
                  additional keys for 'accuracy' and 'equiv_grad_list'.
    """
    if figsize == 'full_width':
        fig = plt.figure(figsize=FIGSIZE_REF)
    elif figsize == 'half_width':
        fig = plt.figure(figsize=FIGSIZE_HALF)
    else:
        fig = plt.figure(figsize=figsize)

    # Retrieve equiv_grad_list and accuracy for plotting
    equiv_grad_list = success_dict.pop('equiv_grad_list')
    accuracy = success_dict.pop('accuracy')

    # Iterate over the solvers in success_dict and plot their success rates
    for hash, success_list in success_dict.items():
        config_str = results.results_utils.hash_to_config_str(hash)
        config = results.results_utils.create_config_from_config_string(config_str=config_str, objective_instance=None)
        
        # Use the hash of the configuration to determine the color
        color = config_str_to_color(str(config))
        linestyle = config_str_to_linestyle(str(config))
        
        label = solver_label(config=config,
                             include_Pk_orth=include_Pk_orth,
                             include_sketch_size=include_sketch_size,
                             include_c_const=include_c_const,
                             include_N_try=include_N_try,
                             include_ensemble=include_ensemble)
        plt.step(equiv_grad_list, success_list, color=color,
                 linestyle=linestyle,label=label, where='post')

    # Add plot title, labels, and legend
    # plt.title(f"Accuracy = {accuracy}")
    plt.xlabel("Equivalent gradient evaluations")
    plt.ylabel("Fraction of problems solved")
    plt.legend(ncol=label_ncol)
    plt.grid(True)
    return fig

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
