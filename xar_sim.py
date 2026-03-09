import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=60)

def is_stationary_ar(phi):
    """
    return true if the ar(p) model is stationary
    """
    phi = np.asarray(phi, dtype=float)
    poly = np.r_[-phi[::-1], 1.0]  # 1 - phi1 z - ... - phip z^p = 0
    roots = np.roots(poly)
    return np.all(np.abs(roots) > 1.0)


def simulate_arp(n, phi, sigma=1.0, x0=None, burn=500, seed=None):
    """
    simulate an ar(p) series:
        x[t] = phi[0]*x[t-1] + ... + phi[p-1]*x[t-p] + eps[t]
    where eps[t] ~ n(0, sigma^2)
    """
    phi = np.asarray(phi, dtype=float)
    p = len(phi)

    if p == 0:
        raise ValueError("phi must have positive length")

    if not is_stationary_ar(phi):
        raise ValueError("phi does not define a stationary ar(p) process")

    total = n + burn + p
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=total)

    x = np.zeros(total, dtype=float)

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
        if len(x0) != p:
            raise ValueError("x0 must have length len(phi)")
        x[:p] = x0

    for t in range(p, total):
        x[t] = np.dot(phi, x[t - p:t][::-1]) + eps[t]

    return x[burn + p:]


def empirical_acf(x, k=5):
    """
    sample autocorrelations at lags 1,...,k
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if k >= n:
        raise ValueError("k must be less than len(x)")

    xc = x - x.mean()
    denom = np.dot(xc, xc)

    acf = np.empty(k, dtype=float)
    for lag in range(1, k + 1):
        acf[lag - 1] = np.dot(xc[:-lag], xc[lag:]) / denom

    return acf


def theoretical_acf_ar(phi, k=5):
    """
    theoretical autocorrelations rho(1),...,rho(k) for a stationary ar(p)
    """
    phi = np.asarray(phi, dtype=float)
    p = len(phi)

    if p == 0:
        raise ValueError("phi must have positive length")

    if not is_stationary_ar(phi):
        raise ValueError("phi does not define a stationary ar(p) process")

    # solve the first p yule-walker equations for rho(1),...,rho(p)
    a = np.zeros((p, p), dtype=float)
    b = np.zeros(p, dtype=float)

    for h in range(1, p + 1):
        a[h - 1, h - 1] = 1.0
        for j in range(1, p + 1):
            m = abs(h - j)
            if m == 0:
                b[h - 1] += phi[j - 1]
            else:
                a[h - 1, m - 1] -= phi[j - 1]

    rho_1_to_p = np.linalg.solve(a, b)

    kmax = max(k, p)
    rho = np.empty(kmax + 1, dtype=float)
    rho[0] = 1.0
    rho[1:p + 1] = rho_1_to_p

    for h in range(p + 1, kmax + 1):
        s = 0.0
        for j in range(1, p + 1):
            s += phi[j - 1] * rho[h - j]
        rho[h] = s

    return rho[1:k + 1]


def compare_acf(x, phi, k=5):
    """
    return lag, empirical acf, theoretical acf, and empirical-theoretical
    """
    emp = empirical_acf(x, k=k)
    theo = theoretical_acf_ar(phi, k=k)
    diff = emp - theo
    lags = np.arange(1, k + 1)
    return np.column_stack((lags, emp, theo, diff))


def main():
    n = 500000
    phi = np.array([0.7, -0.2, 0.1])   # ar(3)
    sigma = 1.0
    k = 5
    seed = 123

    x = simulate_arp(n=n, phi=phi, sigma=sigma, burn=1000, seed=seed)
    table = compare_acf(x, phi, k=k)

    print("phi =", phi)
    print("stationary =", is_stationary_ar(phi))
    print()
    print("first 10 simulated values:")
    print(np.round(x[:10], 4))
    print()
    print(" lag    empirical    theoretical    difference")
    for row in table:
        lag, emp, theo, diff = row
        print(f"{int(lag):4d}   {emp:10.6f}   {theo:11.6f}   {diff:11.6f}")


if __name__ == "__main__":
    main()
