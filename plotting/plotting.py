import matplotlib.pyplot as plt

def plot_loss_vs_iteration(f_vals):
    """
    Plot the loss (function values) vs iteration count.

    :param f_vals: Array of function values over iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(f_vals, linestyle='-', color='b')
    plt.yscale('log')  # Set the vertical axis to log scale
    plt.xlabel('Iteration')
    plt.ylabel('Function value (log scale)')
    plt.title('Loss vs Iteration')
    plt.grid(True, which="both", ls="-")
    plt.show()