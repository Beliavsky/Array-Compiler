import numpy as np


def read_returns_matrix(filename: str):
    """
    read a text file containing an n x m matrix of returns
    """
    x = np.loadtxt(filename, dtype=float)
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.ndim != 2:
        raise ValueError("returns input must be a 2d array")

    n, m = x.shape
    if n < 5:
        raise ValueError("need at least 5 observations")
    if m < 1:
        raise ValueError("need at least 1 series")

    return x


def load_optional_matrix(filename: str):
    try:
        x = np.loadtxt(filename, dtype=float)
    except OSError:
        return np.empty((0, 0))
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def load_optional_vector(filename: str):
    mat = load_optional_matrix(filename)
    if mat.size == 0:
        return np.empty(0)
    if mat.shape[1] == 1:
        return mat[:, 0]
    return np.asarray(mat, dtype=float).reshape(-1)


def garch11_filter(x, omega, alpha, beta):
    """
    conditional variances for a demeaned univariate garch(1,1)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    h = np.empty(n)

    v = np.var(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("variance must be positive")

    h[0] = max(v, omega / max(1.0 - alpha - beta, 1.0e-12))

    for t in range(1, n):
        h[t] = omega + alpha * x[t - 1] ** 2 + beta * h[t - 1]

    return h


def garch11_negloglik(params, x):
    """
    gaussian quasi-negative loglikelihood for demeaned garch(1,1)
    """
    omega, alpha, beta = params

    if omega <= 0.0:
        return np.inf
    if alpha < 0.0 or beta < 0.0:
        return np.inf
    if alpha + beta >= 1.0:
        return np.inf

    h = garch11_filter(x, omega, alpha, beta)

    if np.any(~np.isfinite(h)) or np.any(h <= 0.0):
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
    fit univariate garch(1,1) by grid search in a box
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    omega_lo, omega_hi = omega_bounds
    alpha_lo, alpha_hi = alpha_bounds
    beta_lo, beta_hi = beta_bounds

    if omega_lo <= 0.0 or omega_hi <= omega_lo:
        raise ValueError("invalid omega bounds")
    if alpha_lo < 0.0 or alpha_hi <= alpha_lo:
        raise ValueError("invalid alpha bounds")
    if beta_lo < 0.0 or beta_hi <= beta_lo:
        raise ValueError("invalid beta bounds")

    omega_grid = np.linspace(omega_lo, omega_hi, n_omega)
    alpha_grid = np.linspace(alpha_lo, alpha_hi, n_alpha)
    beta_grid = np.linspace(beta_lo, beta_hi, n_beta)

    best_omega = np.nan
    best_alpha = np.nan
    best_beta = np.nan
    best_nll = np.inf

    for omega in omega_grid:
        for alpha in alpha_grid:
            for beta in beta_grid:
                if alpha + beta >= 0.999:
                    continue

                nll = garch11_negloglik(np.array([omega, alpha, beta]), x)
                if nll < best_nll:
                    best_nll = nll
                    best_omega = omega
                    best_alpha = alpha
                    best_beta = beta

    if not np.isfinite(best_nll):
        raise RuntimeError("no admissible garch parameters found")

    return {
        "omega": best_omega,
        "alpha": best_alpha,
        "beta": best_beta,
        "negloglik": best_nll,
    }


def fit_garch11_refined(
    x,
    n_rounds=3,
    n_omega=12,
    n_alpha=12,
    n_beta=12,
):
    """
    coarse-to-fine grid search for univariate garch(1,1)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    v = np.var(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("variance must be positive")

    omega_lo = max(1.0e-8, 1.0e-4 * v)
    omega_hi = max(5.0e-4 * v, 1.5 * v)
    alpha_lo = 0.01
    alpha_hi = 0.25
    beta_lo = 0.50
    beta_hi = 0.98

    best_nll = np.inf

    for _ in range(n_rounds):
        fit = fit_garch11_grid_search(
            x=x,
            omega_bounds=(omega_lo, omega_hi),
            alpha_bounds=(alpha_lo, alpha_hi),
            beta_bounds=(beta_lo, beta_hi),
            n_omega=n_omega,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )
        best_nll = fit["negloglik"]

        omega_star = fit["omega"]
        alpha_star = fit["alpha"]
        beta_star = fit["beta"]

        omega_half = max(0.25 * (omega_hi - omega_lo), 1.0e-8)
        alpha_half = max(0.25 * (alpha_hi - alpha_lo), 0.01)
        beta_half = max(0.25 * (beta_hi - beta_lo), 0.02)

        omega_lo = max(1.0e-10, omega_star - omega_half)
        omega_hi = omega_star + omega_half
        alpha_lo = max(0.0, alpha_star - alpha_half)
        alpha_hi = min(0.60, alpha_star + alpha_half)
        beta_lo = max(0.0, beta_star - beta_half)
        beta_hi = min(0.999, beta_star + beta_half)

    return {
        "omega": omega_star,
        "alpha": alpha_star,
        "beta": beta_star,
        "negloglik": best_nll,
    }


def ensure_positive_definite_correlation(corr):
    """
    symmetrize and apply a small shrink toward identity
    """
    corr = np.asarray(corr, dtype=float)
    m = corr.shape[0]
    eye = np.eye(m)
    candidate = 0.5 * (corr + corr.T)

    for i in range(m):
        candidate[i, i] = 1.0

    shrink = 1.0e-6
    trial = (1.0 - shrink) * candidate + shrink * eye
    trial = 0.5 * (trial + trial.T)
    for i in range(m):
        trial[i, i] = 1.0
    return trial


def estimate_corr_from_standardized(z):
    """
    sample correlation of standardized residuals
    """
    z = np.asarray(z, dtype=float)
    n, m = z.shape
    corr = np.empty((m, m))

    cov = (z.T @ z) / float(n)
    scale = np.sqrt(np.diag(cov))

    for i in range(m):
        for j in range(m):
            denom = scale[i] * scale[j]
            if denom <= 0.0 or not np.isfinite(denom):
                corr[i, j] = 0.0
            else:
                corr[i, j] = cov[i, j] / denom

    for i in range(m):
        corr[i, i] = 1.0

    return ensure_positive_definite_correlation(corr)


def ccc_negloglik(centered, h, corr):
    """
    gaussian quasi-negative loglikelihood for ccc-garch
    """
    centered = np.asarray(centered, dtype=float)
    h = np.asarray(h, dtype=float)
    corr = np.asarray(corr, dtype=float)

    if h.ndim != 2:
        raise ValueError("h must be a 2d array")

    n, m = centered.shape
    sign, logdet_corr = np.linalg.slogdet(corr)
    if sign <= 0.0:
        return np.inf

    inv_corr = np.linalg.inv(corr)
    total = 0.0

    for t in range(n):
        if np.any(h[t] <= 0.0):
            return np.inf

        z_t = centered[t] / np.sqrt(h[t])
        quad = z_t @ inv_corr @ z_t
        total += 0.5 * (
            m * np.log(2.0 * np.pi)
            + np.sum(np.log(h[t]))
            + logdet_corr
            + quad
        )

    return total


def fit_ccc_garch(
    returns,
    n_rounds=3,
    n_omega=12,
    n_alpha=12,
    n_beta=12,
):
    """
    two-step ccc-garch fit:
    1. fit each marginal as univariate garch(1,1)
    2. estimate constant correlation from standardized residuals
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("returns must be a 2d array")

    n, m = returns.shape
    mu = np.mean(returns, axis=0)
    centered = returns - mu

    omega = np.empty(m)
    alpha = np.empty(m)
    beta = np.empty(m)
    h = np.empty((n, m))

    for j in range(m):
        fit_j = fit_garch11_refined(
            centered[:, j],
            n_rounds=n_rounds,
            n_omega=n_omega,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )
        omega[j] = fit_j["omega"]
        alpha[j] = fit_j["alpha"]
        beta[j] = fit_j["beta"]
        h[:, j] = garch11_filter(centered[:, j], omega[j], alpha[j], beta[j])

    z = centered / np.sqrt(h)
    corr = estimate_corr_from_standardized(z)
    negloglik = ccc_negloglik(centered, h, corr)

    return {
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "corr": corr,
        "negloglik": negloglik,
        "h": h,
        "z": z,
    }


def main():
    infile = "returns_matrix.txt"

    returns = read_returns_matrix(infile)
    fit = fit_ccc_garch(returns)
    mu = fit["mu"]
    omega = fit["omega"]
    alpha = fit["alpha"]
    beta = fit["beta"]
    corr = fit["corr"]
    negloglik = fit["negloglik"]
    h = fit["h"]
    z = fit["z"]

    params_matrix = np.column_stack((omega, alpha, beta))
    np.savetxt("fitted_mean_vector.txt", mu)
    np.savetxt("fitted_garch_params.txt", params_matrix)
    np.savetxt("fitted_corr_matrix.txt", corr)
    np.savetxt("fitted_conditional_variances.txt", h)
    np.savetxt("fitted_standardized_residuals.txt", z)

    print("file", infile)
    print("n, m =", returns.shape)
    print("mean vector")
    print(mu)
    print()
    print("omega")
    print(omega)
    print("alpha")
    print(alpha)
    print("beta")
    print(beta)
    print("alpha + beta")
    print(alpha + beta)
    print()
    print("corr")
    print(corr)
    print()
    print("negloglik", negloglik)
    print()

    true_mu = load_optional_vector("true_mean_vector.txt")
    true_params = load_optional_matrix("true_garch_params.txt")
    true_corr = load_optional_matrix("true_corr_matrix.txt")

    if true_mu.size > 0 and true_mu.shape == mu.shape:
        print("mean comparison")
        print("true  ", true_mu)
        print("fit   ", mu)
        print("abserr", np.abs(mu - true_mu))
        print()

    if true_params.size > 0 and true_params.shape == params_matrix.shape:
        true_omega = true_params[:, 0]
        true_alpha = true_params[:, 1]
        true_beta = true_params[:, 2]
        print("omega comparison")
        print("true  ", true_omega)
        print("fit   ", omega)
        print("abserr", np.abs(omega - true_omega))
        print()
        print("alpha comparison")
        print("true  ", true_alpha)
        print("fit   ", alpha)
        print("abserr", np.abs(alpha - true_alpha))
        print()
        print("beta comparison")
        print("true  ", true_beta)
        print("fit   ", beta)
        print("abserr", np.abs(beta - true_beta))
        print()

    if true_corr.size > 0 and true_corr.shape == corr.shape:
        print("corr comparison")
        print("true")
        print(true_corr)
        print("fit")
        print(corr)
        print("abserr")
        print(np.abs(corr - true_corr))
        print()

    print("wrote fitted_mean_vector.txt")
    print("wrote fitted_garch_params.txt")
    print("wrote fitted_corr_matrix.txt")
    print("wrote fitted_conditional_variances.txt")
    print("wrote fitted_standardized_residuals.txt")


if __name__ == "__main__":
    main()
