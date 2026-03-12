import numpy as np

def simulate_ccc_garch(n, omega, alpha, beta, corr, mu=None, burn=500, seed=None):
    """
    simulate a CCC-GARCH(1,1) process

    r_t = mu + eps_t
    eps_t = D_t L z_t
    h_it = omega_i + alpha_i eps_{i,t-1}^2 + beta_i h_{i,t-1}

    omega, alpha, beta: 1d arrays of length m
    corr: m x m correlation matrix
    mu: 1d array of length m
    """
    rng = np.random.default_rng(seed)

    omega = np.asarray(omega, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    corr = np.asarray(corr, dtype=float)

    m = omega.size

    if mu is None:
        mu = np.zeros(m)
    else:
        mu = np.asarray(mu, dtype=float)

    if alpha.shape != (m,) or beta.shape != (m,) or mu.shape != (m,):
        raise ValueError("omega, alpha, beta, mu must have same length")

    if corr.shape != (m, m):
        raise ValueError("corr must be m x m")

    if np.any(omega <= 0.0):
        raise ValueError("omega must be positive")
    if np.any(alpha < 0.0) or np.any(beta < 0.0):
        raise ValueError("alpha and beta must be nonnegative")
    if np.any(alpha + beta >= 1.0):
        raise ValueError("need alpha + beta < 1 componentwise")

    L = np.linalg.cholesky(corr)

    n_total = n + burn
    h = np.empty((n_total, m))
    eps = np.empty((n_total, m))
    r = np.empty((n_total, m))

    h[0] = omega / (1.0 - alpha - beta)

    z0 = rng.standard_normal(m)
    u0 = L @ z0
    eps[0] = np.sqrt(h[0]) * u0
    r[0] = mu + eps[0]

    for t in range(1, n_total):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        z = rng.standard_normal(m)
        u = L @ z
        eps[t] = np.sqrt(h[t]) * u
        r[t] = mu + eps[t]

    return r[burn:], h[burn:]


if __name__ == "__main__":
    n = 20000
    print("#obs:", n)
    omega = np.array([0.05, 0.03])
    alpha = np.array([0.08, 0.06])
    beta = np.array([0.90, 0.92])

    corr = np.array([
        [1.0, 0.4],
        [0.4, 1.0],
    ])

    r, h = simulate_ccc_garch(
        n=n,
        omega=omega,
        alpha=alpha,
        beta=beta,
        corr=corr,
        seed=123,
    )

    np.savetxt("returns_matrix.txt", r)
    np.savetxt("true_mean_vector.txt", np.zeros_like(omega))
    np.savetxt("true_garch_params.txt", np.column_stack((omega, alpha, beta)))
    np.savetxt("true_corr_matrix.txt", corr)

    print("r.shape", r.shape)
    print("true omega", omega)
    print("true alpha", alpha)
    print("true beta ", beta)
    print("true corr")
    print(corr)
    print()
    print("sample mean")
    print(np.mean(r, axis=0))
    print("sample covariance")
    print(np.cov(r.T))
    print()
    print("wrote returns_matrix.txt")
    print("wrote true_mean_vector.txt")
    print("wrote true_garch_params.txt")
    print("wrote true_corr_matrix.txt")
