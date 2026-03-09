import numpy as np

def is_stationary_ar(phi):
    """
    return true if the ar(p) model is stationary
    """
    phi = np.asarray(phi, dtype=float)
    poly = np.r_[-phi[::-1], 1.0]
    roots = np.roots(poly)
    return np.all(np.abs(roots) > 1.0)


def simulate_arp(n, phi, sigma=1.0, x0=None, burn=500, seed=None):
    """
    simulate an ar(p) series
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
        x[t] = np.dot(phi, x[t-p:t][::-1]) + eps[t]

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
        rho[h] = np.dot(phi, rho[h - np.arange(1, p + 1)])

    return rho[1:k + 1]


def pacf_from_acf(rho):
    """
    given rho[0],...,rho[k], return pacf at lags 1,...,k
    using durbin-levinson recursion
    """
    rho = np.asarray(rho, dtype=float)
    k = len(rho) - 1

    if k < 1:
        return np.array([], dtype=float)

    phi_mat = np.zeros((k + 1, k + 1), dtype=float)
    pacf = np.zeros(k + 1, dtype=float)

    phi_mat[1, 1] = rho[1]
    pacf[1] = rho[1]

    v = 1.0 - rho[1] ** 2

    for m in range(2, k + 1):
        num = rho[m]
        for j in range(1, m):
            num -= phi_mat[m - 1, j] * rho[m - j]

        den = v
        phi_mat[m, m] = num / den
        pacf[m] = phi_mat[m, m]

        for j in range(1, m):
            phi_mat[m, j] = phi_mat[m - 1, j] - phi_mat[m, m] * phi_mat[m - 1, m - j]

        v = v * (1.0 - phi_mat[m, m] ** 2)

    return pacf[1:]


def empirical_pacf(x, k=5):
    """
    sample pacf at lags 1,...,k
    """
    acf = empirical_acf(x, k=k)
    rho = np.empty(k + 1, dtype=float)
    rho[0] = 1.0
    rho[1:] = acf
    return pacf_from_acf(rho)


def theoretical_pacf_ar(phi, k=5):
    """
    theoretical pacf at lags 1,...,k for a stationary ar(p)
    """
    acf = theoretical_acf_ar(phi, k=k)
    rho = np.empty(k + 1, dtype=float)
    rho[0] = 1.0
    rho[1:] = acf
    pacf = pacf_from_acf(rho)

    p = len(phi)
    if k > p:
        pacf[p:] = 0.0

    return pacf


def compare_values(emp, theo):
    """
    return lag, empirical, theoretical, difference
    """
    emp = np.asarray(emp, dtype=float)
    theo = np.asarray(theo, dtype=float)

    if len(emp) != len(theo):
        raise ValueError("emp and theo must have the same length")

    k = len(emp)
    lags = np.arange(1, k + 1)
    diff = emp - theo
    return np.column_stack((lags, emp, theo, diff))


def print_table(title, table):
    print(title)
    print(" lag    empirical    theoretical    difference")
    for row in table:
        lag, emp, theo, diff = row
        print(f"{int(lag):4d}   {emp:10.6f}   {theo:11.6f}   {diff:11.6f}")
    print()


def main():
    n = 5000
    phi = np.array([0.7, -0.2, 0.1])
    sigma = 1.0
    k = 10
    seed = 123
    print("n:", n)
    x = simulate_arp(n=n, phi=phi, sigma=sigma, burn=1000, seed=seed)

    emp_acf = empirical_acf(x, k=k)
    theo_acf = theoretical_acf_ar(phi, k=k)
    acf_table = compare_values(emp_acf, theo_acf)

    emp_pacf = empirical_pacf(x, k=k)
    theo_pacf = theoretical_pacf_ar(phi, k=k)
    pacf_table = compare_values(emp_pacf, theo_pacf)

    print("phi =", phi)
    print("stationary =", is_stationary_ar(phi))
    print()
    print("first 10 simulated values:")
    print(np.round(x[:10], 4))
    print()

    print_table("acf", acf_table)
    print_table("pacf", pacf_table)


if __name__ == "__main__":
    main()
