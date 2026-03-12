import numpy as np

def simulate_garch11(n, omega, alpha, beta, mu=0.0, burn=1000, seed=123):
    """
    simulate garch(1,1) returns
    """
    if omega <= 0.0:
        raise ValueError("omega must be > 0")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be >= 0")
    if alpha + beta >= 1.0:
        raise ValueError("need alpha + beta < 1")

    rng = np.random.default_rng(seed)

    m = n + burn
    z = rng.standard_normal(m)
    h = np.empty(m)
    eps = np.empty(m)
    r = np.empty(m)

    h[0] = omega / (1.0 - alpha - beta)
    eps[0] = np.sqrt(h[0]) * z[0]
    r[0] = mu + eps[0]

    for t in range(1, m):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        eps[t] = np.sqrt(h[t]) * z[t]
        r[t] = mu + eps[t]

    return r[burn:], h[burn:]


def main():
    n = 2000
    omega = 0.05
    alpha = 0.08
    beta = 0.90
    mu = 0.0
    burn = 1000
    seed = 123
    outfile = "returns.txt"

    r, h = simulate_garch11(
        n=n,
        omega=omega,
        alpha=alpha,
        beta=beta,
        mu=mu,
        burn=burn,
        seed=seed,
    )

    np.savetxt(outfile, r, fmt="%.10f")

    print("wrote", outfile)
    print("n", n)
    print("omega", omega)
    print("alpha", alpha)
    print("beta", beta)
    print("mu", mu)
    print("sample mean", np.mean(r))
    print("sample variance", np.var(r))


if __name__ == "__main__":
    main()
