
"""Solve NNLS problem via a simple projected gradient method.

Usefull when the matrix A is very large but has special structure
that allows it to be implemented via an operator (e.g convolution).
Ax_fn should be a function which takes a single argument, x, and returns
the matrix vector product Ax. ATx_fn should take x and return the
the matrix vector product A^Tx.

Step sizes are currently chosen by backtracking line search, but
we can add lipschitz constant calculations to get the optimal stepsize.
"""
function projected_gradient(Ax_fn::Function,
                            ATx_fn::Function,
                            x_dim::Integer,
                            B::AbstractMatrix;
                            alpha=0.1,
                            beta=0.8,
                            init_step_size=0.1,
                            max_iter=10000,
                            eps=1e-8,
                            kwargs...)
    function loss(X)
        return norm(Ax_fn(X) - B)^2
    end

    X = get(kwargs, :X_init, nothing)
    (N, P) = size(B)
    if X == nothing
        X = rand(x_dim, P)
    end

    loss_cur = loss(X)
    step_size = init_step_size
    for iter in 1:max_iter
        grad = ATx_fn(Ax_fn(X)) - ATx_fn(B)
        X_prime = max.(0, X - step_size * grad)

        # Backtracking line search, roughly following
        # Convex Optimization (Boyd and Vandenberghe) page 464
        new_loss = loss(X_prime)
        grad_norm = norm(grad) ^ 2
        while new_loss > loss_cur - sum(grad .* (X - X_prime)) + 1 / (2 * step_size) * norm(X - X_prime)
            step_size = beta * step_size
            X_prime = max.(0, X - step_size * grad)
            new_loss = loss(X_prime)
        end
        X = X_prime
        loss_cur = new_loss

        # Convergence is calculated using the max-norm of the `projected
        # gradient,` i.e the part of the gradient which would affect entries
        # of X where the constraints are not active.
        grad[X .== 0] = min.(0, grad[X .== 0])
        if maximum(abs.(grad)) < eps
            break
        end
    end
    @warn "Tolerance not reached after maximum iterations."
    return X
end
