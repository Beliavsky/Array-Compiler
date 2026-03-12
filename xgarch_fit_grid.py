import numpy as np


def read_returns(filename):
    """
    read one-column text file of returns
    """
    x = np.loadtxt(filename, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("need at least 2 returns")
    return x


def garch11_negloglik(params, x):
    """
    gaussian quasi-negative loglikelihood for garch(1,1)
    """
    omega, alpha, beta = params

    if omega <= 0.0:
        return np.inf
    if alpha < 0.0 or beta < 0.0:
        return np.inf
    if alpha + beta >= 1.0:
        return np.inf

    n = x.size
    h = np.empty(n)

    v = np.var(x)
    if not np.isfinite(v) or v <= 0.0:
        return np.inf

    h[0] = max(v, omega / max(1.0 - alpha - beta, 1.0e-12))

    for t in range(1, n):
        h[t] = omega + alpha * x[t - 1] ** 2 + beta * h[t - 1]
        if not np.isfinite(h[t]) or h[t] <= 0.0:
            return np.inf

    return 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + x ** 2 / h)


def fit_garch11_grid_search(
    x,
    omega_bounds,
    alpha_bounds,
    beta_bounds,
    n_omega,
    n_alpha,
    n_beta,
):
    """
    fit garch(1,1) by grid search in a box
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.mean(x)

    if np.var(x) <= 0.0:
        raise ValueError("variance must be positive")

    if n_omega < 1 or n_alpha < 1 or n_beta < 1:
        raise ValueError("grid sizes must be positive integers")

    omega_lo, omega_hi = omega_bounds
    alpha_lo, alpha_hi = alpha_bounds
    beta_lo, beta_hi = beta_bounds

    if omega_lo >= omega_hi or alpha_lo >= alpha_hi or beta_lo >= beta_hi:
        raise ValueError("each lower bound must be smaller than upper bound")

    omega_grid = np.linspace(omega_lo, omega_hi, n_omega)
    alpha_grid = np.linspace(alpha_lo, alpha_hi, n_alpha)
    beta_grid = np.linspace(beta_lo, beta_hi, n_beta)

    best_params = None
    best_nll = np.inf
    n_eval = 0
    n_admissible = 0

    for omega in omega_grid:
        for alpha in alpha_grid:
            for beta in beta_grid:
                n_eval += 1

                if alpha + beta >= 1.0:
                    continue

                n_admissible += 1
                params = np.array([omega, alpha, beta], dtype=float)
                nll = garch11_negloglik(params, x)

                if nll < best_nll:
                    best_nll = nll
                    best_params = params.copy()

    if best_params is None:
        raise RuntimeError("no admissible grid points found")

    return {
        "omega": best_params[0],
        "alpha": best_params[1],
        "beta": best_params[2],
        "negloglik": best_nll,
        "n_eval": n_eval,
        "n_admissible": n_admissible,
    }


def main():
    infile = "returns.txt"

    x = read_returns(infile)

    x0 = x - np.mean(x)
    v = np.var(x0)
    n_omega = 50
    n_alpha = 50
    n_beta = 10
    print("n_omega, n_alpha, n_beta =", n_omega, n_alpha, n_beta)
    fit = fit_garch11_grid_search(
        x=x,
        omega_bounds=(1.0e-6, v),
        alpha_bounds=(0.001, 0.25),
        beta_bounds=(0.50, 0.999),
        n_omega=n_omega,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    print("file", infile)
    print("n", x.size)
    print("mean", np.mean(x))
    print("variance", np.var(x))
    print()
    print("estimated omega", fit["omega"])
    print("estimated alpha", fit["alpha"])
    print("estimated beta ", fit["beta"])
    print("alpha + beta", fit["alpha"] + fit["beta"])
    print("negloglik", fit["negloglik"])
    print()
    print("grid points checked", fit["n_eval"])
    print("admissible points", fit["n_admissible"])


if __name__ == "__main__":
    main()
