import matplotlib.pyplot as plt
from solvers.utils import SolverOutput

def plot_loss_vs_iteration(solver_outputs, labels=None):
    """
    Plot the loss (function values) vs iteration count for multiple solvers.

    :param solver_outputs: List of SolverOutput instances.
    :param labels: List of labels for each solver output.
    """
    plt.figure(figsize=(10, 6))
    
    if labels is None:
        labels = [f"Solver {i+1}" for i in range(len(solver_outputs))]

    for solver_output, label in zip(solver_outputs, labels):
        plt.plot(solver_output.f_vals, linestyle='-', label=label)
    
    plt.yscale('log')  # Set the vertical axis to log scale
    plt.xlabel('Iteration')
    plt.ylabel('Function value (log scale)')
    plt.title('Loss vs Iteration Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()