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

    if not is_invertible_ma(ma):
        raise ValueError("ma does not define an invertible process")

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
            ar_part = np.dot(ar, x[t-p:t][::-1])

        ma_part = eps[t]
        if q > 0:
            ma_part += np.dot(ma, eps[t-q:t][::-1])

        x[t] = ar_part + ma_part

    return x[burn + r:]


def empirical_acf(x, k):
    """
    sample acf at lags 1,...,k
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

    v = max(1.0 - rho[1] ** 2, 1.0e-12)

    for m in range(2, k + 1):
        num = rho[m]
        for j in range(1, m):
            num -= phi[m - 1, j] * rho[m - j]

        phi[m, m] = num / v
        pacf[m] = phi[m, m]

        for j in range(1, m):
            phi[m, j] = phi[m - 1, j] - phi[m, m] * phi[m - 1, m - j]

        v = max(v * (1.0 - phi[m, m] ** 2), 1.0e-12)

    return pacf[1:]


def empirical_pacf(x, k):
    """
    sample pacf at lags 1,...,k
    """
    acf = empirical_acf(x, k=k)
    rho = np.empty(k + 1, dtype=float)
    rho[0] = 1.0
    rho[1:] = acf
    return pacf_from_acf(rho)


def impulse_response_arma(ar, ma, npsi):
    """
    return psi[0],...,psi[npsi-1] where

    x[t] = sum_{j>=0} psi[j] * eps[t-j]
    """
    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)

    p = len(ar)
    q = len(ma)

    psi = np.zeros(npsi, dtype=float)
    psi[0] = 1.0

    for j in range(1, npsi):
        s = 0.0

        if j <= q:
            s += ma[j - 1]

        m = min(p, j)
        for i in range(1, m + 1):
            s += ar[i - 1] * psi[j - i]

        psi[j] = s

    return psi


def theoretical_acf_arma(ar, ma, k, npsi=None):
    """
    approximate theoretical acf at lags 1,...,k using truncated psi weights
    """
    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)

    if not is_stationary_ar(ar):
        raise ValueError("ar does not define a stationary process")

    if not is_invertible_ma(ma):
        raise ValueError("ma does not define an invertible process")

    if npsi is None:
        npsi = max(500, 50 * (k + 1), 200 * (max(len(ar), len(ma)) + 1))

    psi = impulse_response_arma(ar, ma, npsi=npsi)

    gamma = np.empty(k + 1, dtype=float)
    for h in range(k + 1):
        gamma[h] = np.dot(psi[:npsi - h], psi[h:])

    return gamma[1:] / gamma[0]


def theoretical_pacf_arma(ar, ma, k, npsi=None):
    """
    theoretical pacf at lags 1,...,k
    """
    acf = theoretical_acf_arma(ar, ma, k=k, npsi=npsi)
    rho = np.empty(k + 1, dtype=float)
    rho[0] = 1.0
    rho[1:] = acf
    return pacf_from_acf(rho)


def yule_walker_ar_start(acf_emp, p):
    """
    ar(p) start from yule-walker using the first p sample acf values
    """
    if p == 0:
        return np.array([], dtype=float)

    rho = np.empty(p + 1, dtype=float)
    rho[0] = 1.0
    rho[1:] = acf_emp[:p]

    rmat = np.empty((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            rmat[i, j] = rho[abs(i - j)]

    rhs = rho[1:]

    try:
        ar = np.linalg.solve(rmat, rhs)
    except np.linalg.LinAlgError:
        ar = np.linalg.lstsq(rmat, rhs, rcond=None)[0]

    ar = np.clip(ar, -0.9, 0.9)

    if not is_stationary_ar(ar):
        ar[:] = 0.0

    return ar


def split_params(params, p, q):
    """
    split parameter vector into ar and ma parts
    """
    params = np.asarray(params, dtype=float)
    ar = params[:p]
    ma = params[p:p + q]
    return ar, ma


def arma_match_objective(params, acf_emp, pacf_emp, p, q,
                         mode="both", pacf_weight=0.5, npsi=None):
    """
    objective for matching empirical and theoretical acf/pacf
    """
    ar, ma = split_params(params, p, q)

    if np.any(np.abs(params) >= 0.999):
        return 1.0e30

    if not is_stationary_ar(ar):
        return 1.0e30

    if not is_invertible_ma(ma):
        return 1.0e30

    k = len(acf_emp)

    try:
        acf_theo = theoretical_acf_arma(ar, ma, k=k, npsi=npsi)
        pacf_theo = theoretical_pacf_arma(ar, ma, k=k, npsi=npsi)
    except Exception:
        return 1.0e30

    w = 1.0 / np.arange(1, k + 1, dtype=float)

    val = 0.0

    if mode in ("acf", "both"):
        d = acf_emp - acf_theo
        val += np.dot(w, d * d)

    if mode in ("pacf", "both"):
        d = pacf_emp - pacf_theo
        val += pacf_weight * np.dot(w, d * d)

    return val


def local_coordinate_search(start, acf_emp, pacf_emp, p, q,
                            mode="both", pacf_weight=0.5,
                            step0=0.3, step_min=1.0e-4, shrink=0.5,
                            max_rounds=100, bound=0.98,
                            npsi=None, rng=None):
    """
    simple pure-numpy local search
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.clip(np.asarray(start, dtype=float).copy(), -bound, bound)
    fx = arma_match_objective(x, acf_emp, pacf_emp, p, q,
                              mode=mode, pacf_weight=pacf_weight,
                              npsi=npsi)

    d = len(x)
    step = step0
    rounds = 0

    while step > step_min and rounds < max_rounds:
        rounds += 1
        improved = False

        for i in range(d):
            for sign in (-1.0, 1.0):
                cand = x.copy()
                cand[i] = np.clip(cand[i] + sign * step, -bound, bound)

                fc = arma_match_objective(cand, acf_emp, pacf_emp, p, q,
                                          mode=mode, pacf_weight=pacf_weight,
                                          npsi=npsi)

                if fc < fx:
                    x = cand
                    fx = fc
                    improved = True

        if not improved and d > 0:
            for _ in range(2 * d):
                direction = rng.normal(size=d)
                norm = np.linalg.norm(direction)
                if norm > 0.0:
                    direction /= norm

                cand = np.clip(x + step * direction, -bound, bound)

                fc = arma_match_objective(cand, acf_emp, pacf_emp, p, q,
                                          mode=mode, pacf_weight=pacf_weight,
                                          npsi=npsi)

                if fc < fx:
                    x = cand
                    fx = fc
                    improved = True
                    break

        if not improved:
            step *= shrink

    return x, fx


def fit_arma_initial_numpy(x, p, q, k,
                           mode="both",
                           pacf_weight=0.5,
                           n_starts=25,
                           step0=0.3,
                           step_min=1.0e-4,
                           shrink=0.5,
                           max_rounds=100,
                           bound=0.98,
                           npsi=None,
                           seed=None):
    """
    fit an initial arma(p, q) guess from the first k empirical acf/pacf values
    using only numpy

    returns a dictionary with ar, ma, empirical/theoretical acf/pacf, and
    the objective value
    """
    x = np.asarray(x, dtype=float)

    if p < 0 or q < 0:
        raise ValueError("p and q must be nonnegative")

    if p == 0 and q == 0:
        raise ValueError("at least one of p, q must be positive")

    acf_emp = empirical_acf(x, k=k)
    pacf_emp = empirical_pacf(x, k=k)

    d = p + q
    rng = np.random.default_rng(seed)

    starts = []

    ar0 = yule_walker_ar_start(acf_emp, p)
    ma0 = np.zeros(q, dtype=float)
    starts.append(np.r_[ar0, ma0])

    starts.append(np.zeros(d, dtype=float))

    for _ in range(max(0, n_starts - len(starts))):
        s = rng.uniform(low=-0.6, high=0.6, size=d)
        starts.append(s)

    best_params = None
    best_obj = 1.0e300

    for s in starts:
        params, obj = local_coordinate_search(
            s,
            acf_emp=acf_emp,
            pacf_emp=pacf_emp,
            p=p,
            q=q,
            mode=mode,
            pacf_weight=pacf_weight,
            step0=step0,
            step_min=step_min,
            shrink=shrink,
            max_rounds=max_rounds,
            bound=bound,
            npsi=npsi,
            rng=rng
        )

        if obj < best_obj:
            best_obj = obj
            best_params = params

    ar_hat, ma_hat = split_params(best_params, p, q)
    acf_theo = theoretical_acf_arma(ar_hat, ma_hat, k=k, npsi=npsi)
    pacf_theo = theoretical_pacf_arma(ar_hat, ma_hat, k=k, npsi=npsi)

    return {
        "ar": ar_hat,
        "ma": ma_hat,
        "objective": best_obj,
        "acf_emp": acf_emp,
        "acf_theo": acf_theo,
        "pacf_emp": pacf_emp,
        "pacf_theo": pacf_theo
    }


def compare_values(emp, theo):
    """
    return lag, empirical, theoretical, difference
    """
    emp = np.asarray(emp, dtype=float)
    theo = np.asarray(theo, dtype=float)
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
    n = 4000
    ar_true = np.array([0.60, -0.25])
    ma_true = np.array([0.50, 0.20])
    sigma = 1.0
    k = 10
    seed = 123

    x = simulate_arma(n=n, ar=ar_true, ma=ma_true, sigma=sigma, burn=1500, seed=seed)

    res = fit_arma_initial_numpy(
        x,
        p=2,
        q=2,
        k=k,
        mode="both",
        pacf_weight=0.5,
        n_starts=30,
        step0=0.25,
        step_min=1.0e-4,
        shrink=0.5,
        max_rounds=10,
        bound=0.98,
        npsi=1500,
        seed=seed
    )

    acf_table = compare_values(res["acf_emp"], res["acf_theo"])
    pacf_table = compare_values(res["pacf_emp"], res["pacf_theo"])

    print("true ar =", np.round(ar_true, 6))
    print("true ma =", np.round(ma_true, 6))
    print()

    print("initial guess from acf/pacf matching")
    print("ar_hat =", np.round(res["ar"], 6))
    print("ma_hat =", np.round(res["ma"], 6))
    print("objective =", float(np.round(res["objective"], 8)))
    print()

    print_table("acf", acf_table)
    print_table("pacf", pacf_table)


if __name__ == "__main__":
    main()
