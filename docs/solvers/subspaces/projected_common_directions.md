# ```projected_common_directions.md```

This script is largely based on the work from https://doi.org/10.1007/s12532-022-00219-z (Lee, Wang, and Lin, 2022). There is another module in this repo, common_directions.py, reflecting the algorithms proposed in that paper closely. This module adapts that with some changes.

Namely, here we implement a version of that algorithm that never uses full gradient information.
We also seek to make this version of the algorithm more general, including variants that do/do not use Hessian information, etc.
The common thread to all of these methods is that THE FULL GRADIENT IS NEVER USED IN ITSELF IN THE ALGORITHM.

There are broadly two ideas for how to "define" projected gradients. One could use randomisation or refrain from it. Suppose that at each iteration our $P_k$ subspace matrix construction makes use of gradient information (in some form) at the previous m iterates.

## Some notation

We use $P_k$ to denote matrices whose column range describes the subspace on which we solve the subproblem at each iteration. These are **not**, in general, orthonormal.

We therefore use $Q_k$ to denote matrices with the same column space as $P_k$, but which are orthonormal (i.e., $Q_k^\top Q_k = I$). This is what we use in computations in the main optimiser loop.

We use $B_k$ to refer to either the Hessian (if the relevant Boolean attribute of the solver class is set to true) or some other approximation to the Hessian (think quasi-Newton). If the user sets $B_k = I$, Newton-like directions become steepest descent-like directions instead.
$$
B_k p_k = - \tilde{\nabla}{f}(x_k)
$$

## Projected gradients

### Without randomisation

Without using randomisation, we can, in constructing P_k, use a projection of the gradient at the current iterate using the orthogonalised subspace basis matrix of iteration (k-1).
$$
\begin{align}
P_k = Q_k R_k \quad \forall k \in \mathbb{N} \text{ (QR decomposition)} \\
\tilde{\nabla}{f}(x_{k+1}) := Q_k Q_k^\top \nabla{f}(x_{k+1}) \\
P_{k+1} = P_{k+1}\left(\tilde{\nabla}{f}(x_{k+1})\right)
\end{align}
$$

We could store $m$ past projected gradients as is, which seems more sensible. A (seemingly much more compute-intensive) alternative would be to recompute $m$ projected gradients at each iteration, using only the previously known (k-1)th basis $Q_{k-1}$. This would scale, for each iteration, the number of directional derivatives to be computed up by a factor of m. For this reason I stick with the former option, at least to begin with.

### With randomisation

While using randomisation, we may use the ideas above while, instead, choosing to define the "projected" gradients by
$$
\tilde{\nabla}{f}(x_k) = S_k S_k^\top \nabla{f}(x_k),
$$
where $S_k$ is a (**tall**) sketching matrix from an appropriately chosen random ensemble. In this case, the word projection takes on its less strict sense from randomised NLA (not the classical meaning from linear algebra/functional analysis, where exact idempotence is required).