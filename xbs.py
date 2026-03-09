import math


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
) -> float:
    """Return the Black-Scholes price of a European call or put."""
    if spot <= 0.0:
        raise ValueError("spot must be > 0")
    if strike <= 0.0:
        raise ValueError("strike must be > 0")
    if volatility <= 0.0:
        raise ValueError("volatility must be > 0")
    if time_to_maturity <= 0.0:
        raise ValueError("time_to_maturity must be > 0")

    option_kind = option_type.strip().lower()
    if option_kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_strike = strike * math.exp(-rate * time_to_maturity)

    if option_kind == "call":
        return spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    return discounted_strike * normal_cdf(-d2) - spot * normal_cdf(-d1)


def run_example() -> None:
    """Price one call and one put and verify put-call parity."""
    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.20
    time_to_maturity = 1.0

    call_price = black_scholes_price(spot, strike, rate, volatility, time_to_maturity, "call")
    put_price = black_scholes_price(spot, strike, rate, volatility, time_to_maturity, "put")

    parity_left = call_price - put_price
    parity_right = spot - strike * math.exp(-rate * time_to_maturity)
    parity_error = abs(parity_left - parity_right)

    print(f"call price: {call_price:.10f}")
    print(f"put price:  {put_price:.10f}")
    print(f"parity lhs: {parity_left:.10f}")
    print(f"parity rhs: {parity_right:.10f}")
    print(f"abs error:  {parity_error:.10e}")

    tolerance = 1.0e-10
    if parity_error <= tolerance:
        print("put-call parity check passed")
    else:
        raise AssertionError("put-call parity check failed")


if __name__ == "__main__":
    run_example()
