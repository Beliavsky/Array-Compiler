import numpy as np

def is_stationary_ar(ar):
    """
    return true if the ar polynomial has roots outside the unit circle
    """
    ar = np.asarray(ar, dtype=float)
    if len(ar) == 0:
        return True
    poly = np.r_[-ar[::-1], 1.0]
    roots = np.roots(poly)
    return np.all(np.abs(roots) > 1.0)


def is_invertible_ma(ma):
    """
    return true if the ma polynomial has roots outside the unit circle
    """
    ma = np.asarray(ma, dtype=float)
    if len(ma) == 0:
        return True
    poly = np.r_[ma[::-1], 1.0]
    roots = np.roots(poly)
    return np.all(np.abs(roots) > 1.0)


def simulate_arma(n, ar=None, ma=None, sigma=1.0, burn=1000, seed=None):
    """
    simulate an arma(p, q) series

    x[t] = sum ar[i-1] * x[t-i] + eps[t] + sum ma[j-1] * eps[t-j]
    """
    if ar is None:
        ar = np.array([], dtype=float)
    if ma is None:
        ma = np.array([], dtype=float)

    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)

    if not is_stationary_ar(ar):
        raise ValueError("ar does not define a stationary process")

    p = len(ar)
    q = len(ma)
    r = max(p, q)

    total = n + burn + r
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=total)
    x = np.zeros(total, dtype=float)

    for t in range(r, total):
        ar_part = 0.0
        if p > 0:
            ar_part = np.dot(ar, x[t - p:t][::-1])

        ma_part = eps[t]
        if q > 0:
            ma_part += np.dot(ma, eps[t - q:t][::-1])

        x[t] = ar_part + ma_part

    return x[burn + r:]


def empirical_acf(x, k=10):
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


def pacf_from_acf(rho):
    """
    given rho[0],...,rho[k], return pacf at lags 1,...,k
    using durbin-levinson recursion
    """
    rho = np.asarray(rho, dtype=float)
    k = len(rho) - 1

    if k < 1:
        return np.array([], dtype=float)

    phi = np.zeros((k + 1, k + 1), dtype=float)
    pacf = np.zeros(k + 1, dtype=float)

    phi[1, 1] = rho[1]
    pacf[1] = rho[1]

    v = 1.0 - rho[1] ** 2

    for m in range(2, k + 1):
        num = rho[m]
        for j in range(1, m):
            num -= phi[m - 1, j] * rho[m - j]

        phi[m, m] = num / v
        pacf[m] = phi[m, m]

        for j in range(1, m):
            phi[m, j] = phi[m - 1, j] - phi[m, m] * phi[m - 1, m - j]

        v *= 1.0 - phi[m, m] ** 2

    return pacf[1:]


def empirical_pacf(x, k=10):
    """
    sample pacf at lags 1,...,k
    """
    acf = empirical_acf(x, k=k)
    rho = np.empty(k + 1, dtype=float)
    rho[0] = 1.0
    rho[1:] = acf
    return pacf_from_acf(rho)


def impulse_response_arma(ar=None, ma=None, n=50):
    """
    return psi[0],...,psi[n-1] where
    x[t] = sum_{j>=0} psi[j] * eps[t-j]
    """
    if ar is None:
        ar = np.array([], dtype=float)
    if ma is None:
        ma = np.array([], dtype=float)

    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)

    p = len(ar)
    q = len(ma)

    psi = np.zeros(n, dtype=float)
    psi[0] = 1.0

    for j in range(1, n):
        s = 0.0

        if j <= q:
            s += ma[j - 1]

        m = min(p, j)
        for i in range(1, m + 1):
            s += ar[i - 1] * psi[j - i]

        psi[j] = s

    return psi


def theoretical_acovf_arma(ar=None, ma=None, sigma=1.0, k=10):
    """
    return theoretical autocovariances gamma(0),...,gamma(k)
    for a stationary arma(p, q)

    uses exact linear equations for the first max(p, q) lags,
    then the ar recursion for higher lags
    """
    if ar is None:
        ar = np.array([], dtype=float)
    if ma is None:
        ma = np.array([], dtype=float)

    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)

    if not is_stationary_ar(ar):
        raise ValueError("ar does not define a stationary process")

    p = len(ar)
    q = len(ma)
    m = max(p, q)

    theta = np.empty(q + 1, dtype=float)
    theta[0] = 1.0
    if q > 0:
        theta[1:] = ma

    psi = impulse_response_arma(ar=ar, ma=ma, n=q + 1)

    a = np.zeros((m + 1, m + 1), dtype=float)
    b = np.zeros(m + 1, dtype=float)

    for h in range(m + 1):
        a[h, h] = 1.0

        for i in range(1, p + 1):
            j = abs(h - i)
            a[h, j] -= ar[i - 1]

        s = 0.0
        for j in range(h, q + 1):
            s += theta[j] * psi[j - h]
        b[h] = sigma ** 2 * s

    gamma0_to_m = np.linalg.solve(a, b)

    gamma = np.zeros(k + 1, dtype=float)
    gamma[:m + 1] = gamma0_to_m[:min(m, k) + 1]

    for h in range(m + 1, k + 1):
        if p == 0:
            gamma[h] = 0.0
        else:
            gamma[h] = np.dot(ar, gamma[h - np.arange(1, p + 1)])

    return gamma


def theoretical_acf_arma(ar=None, ma=None, sigma=1.0, k=10):
    """
    return theoretical autocorrelations rho(1),...,rho(k)
    """
    gamma = theoretical_acovf_arma(ar=ar, ma=ma, sigma=sigma, k=k)
    return gamma[1:] / gamma[0]


def theoretical_pacf_arma(ar=None, ma=None, sigma=1.0, k=10):
    """
    return theoretical pacf at lags 1,...,k
    """
    acf = theoretical_acf_arma(ar=ar, ma=ma, sigma=sigma, k=k)
    rho = np.empty(k + 1, dtype=float)
    rho[0] = 1.0
    rho[1:] = acf
    return pacf_from_acf(rho)


def compare_values(emp, theo):
    """
    return lag, empirical, theoretical, difference
    """
    emp = np.asarray(emp, dtype=float)
    theo = np.asarray(theo, dtype=float)

    if len(emp) != len(theo):
        raise ValueError("emp and theo must have the same length")

    lags = np.arange(1, len(emp) + 1)
    diff = emp - theo
    return np.column_stack((lags, emp, theo, diff))


def print_table(title, table):
    print(title)
    print(" lag    empirical    theoretical    difference")
    for lag, emp, theo, diff in table:
        print(f"{int(lag):4d}   {emp:10.6f}   {theo:11.6f}   {diff:11.6f}")
    print()


def main():
    n = 5000
    ar = np.array([0.6, -0.3])     # ar(1)
    ma = np.array([0.1])     # ma(1)
    sigma = 1.0
    k = 10
    seed = 123
    print("n:", n)
    x = simulate_arma(n=n, ar=ar, ma=ma, sigma=sigma, burn=1000, seed=seed)

    emp_acf = empirical_acf(x, k=k)
    theo_acf = theoretical_acf_arma(ar=ar, ma=ma, sigma=sigma, k=k)
    acf_table = compare_values(emp_acf, theo_acf)

    emp_pacf = empirical_pacf(x, k=k)
    theo_pacf = theoretical_pacf_arma(ar=ar, ma=ma, sigma=sigma, k=k)
    pacf_table = compare_values(emp_pacf, theo_pacf)

    print("ar =", ar)
    print("ma =", ma)
    print("stationary ar part =", is_stationary_ar(ar))
    print("invertible ma part =", is_invertible_ma(ma))
    print()
    print("first 10 simulated values:")
    print(np.round(x[:10], 4))
    print()

    print_table("acf", acf_table)
    print_table("pacf", pacf_table)


if __name__ == "__main__":
    main()
