# ```projected_common_directions.md```

## Some notation

We use $P_k$ to denote matrices whose column range is contained in the subspace to which we reduce the problem. These columns are constructed using problem information (e.g., iterates, projected gradients of the objective, ...). We may refer to the orthogonalised basis as $\hat{Q}_k$, through the QR factorisation
$$
P_k = \hat{Q}_k \hat{R}_k \in \mathbb{R}^{n \times m_1}, \quad \forall k \in \mathbb{N}
$$

However, in general we may also, at each iteration, append a few randomised directions, orthogonal to those of $\hat{Q}_k$. We may refer to their orthogonalised basis matrix as $\tilde{Q}_k$. Note that
$$
\tilde{Q}_k \in \mathbb{R}^{n \times m_2}, \quad \forall k \in \mathbb{N}
$$
As mentioned, this is constructed so that
$$
\tilde{Q}_k^\top \hat{Q}_k = 0, \quad \forall k \in \mathbb{N}.
$$

Note that thus the entire subspace orthonormal basis matrix is
$$
\begin{align}
Q_k = \left[ \hat{Q}_k, \, \tilde{Q}_k \right] \in \mathbb{R}^{n \times m}
\end{align}
$$
Note that $m (= m_1 + m_2)$ is the dimension of the matrices used to compute projected gradients as well (whether deterministic or randomised ones).


We use $B_k$ to refer to either the Hessian (if the relevant Boolean attribute of the solver class is set to true) or some other approximation to the Hessian (think quasi-Newton). If the user sets $B_k = I$, Newton-like directions become steepest descent-like directions instead.
$$
B_k p_k = - \tilde{\nabla}{f}(x_k)
$$

## Projected gradients

### Without randomisation

#### Without reprojection

Without using randomisation, we can, in constructing $P_k$, use a projection of the gradient at the current iterate using the orthogonalised subspace basis matrix of iteration $(k-1)$.
$$
\begin{align}
P_k = \hat{Q}_k \hat{R}_k \quad \forall k \in \mathbb{N} \text{ (QR decomposition)} \\
\tilde{\nabla}{f}(x_k) := Q_{k-1} Q_{k-1}^\top \nabla{f}(x_{k}) \\
\end{align}
$$

We could store $m$ past projected gradients as is, which seems sensible.

#### With reprojection

A (seemingly much more compute-intensive) alternative (which is not implemented) would be to recompute $m$ projected gradients at each iteration, using only the previously known (k-1)th basis $Q_{k-1}$. This would scale, for each iteration, the number of directional derivatives to be computed up by a factor of $m$.

Yet another, not so expensive, alternative, is to reproject gradients, after the "fact", so that the projection matrix and iterate **have the same iteration subscript**. That is, at iteration $k$, we do:
$$
\begin{align}
\tilde{\nabla}{f}(x_k) &:= Q_{k-1} Q_{k-1}^\top \nabla{f}(x_k), \\
\tilde{\nabla}{f}(x_{k-1}) &:= Q_{k-1} Q_{k-1}^\top \nabla{f}(x_{k-1}). \quad \text{(reassignment)}
\end{align}
$$
In this manner, all gradients except for the current one are projected, after the fact (i.e., for use in subsequent subspace constructions), with the subspace basis matrix used in their corresponding iteration.

**Note that**, if we only store a **single** projected gradient in $P_k$ at each iteration, none of this reassignment takes place, since each gradient is discarded by the time we're considering the next iterate.

### With randomisation

While using randomisation, we may use the ideas above while, instead, choosing to define the "projected" gradients by
$$
\tilde{\nabla}{f}(x_k) = S_k S_k^\top \nabla{f}(x_k),
$$
where
$$
S_k \in \mathbb{R}^{n \times m}
$$
is a (**tall**) sketching matrix from an appropriately chosen random ensemble. In this case, the word projection takes on its less strict sense from randomised NLA (not the classical meaning from linear algebra/functional analysis, where exact idempotence is required).

In this case there would seem to be little need to require reprojection of gradients, since projection matrices are randomised.

## Subspace basis constructions

Generically, we use $m_1$ directions derived from problem information, such as past iterates, projected gradients, etc.

In general, we also have $m_2$ (which may be zero) directions, orthogonal to the aforementioned $m_1$ directions, appended to extend the subspace with some degree of randomness.