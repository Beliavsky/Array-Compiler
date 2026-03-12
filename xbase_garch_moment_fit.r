# xgarch_sim.R
# simulate returns from a garch(1,1) model using base r only
# write the simulated returns to a file
# print:
#   1) acf of squared returns
#   2) moments of returns
#   3) moment-based estimates of garch(1,1) parameters
#   4) normality tests for returns and standardized residuals
#
# model:
#   r_t = mu + eps_t
#   eps_t = sqrt(sigma2_t) * z_t,   z_t ~ N(0,1)
#   sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

simulate_garch11 <- function(n, omega, alpha, beta, mu = 0, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  if (n < 11) {
    stop("n must be at least 11")
  }
  if (omega <= 0) {
    stop("omega must be > 0")
  }
  if (alpha < 0 || beta < 0) {
    stop("alpha and beta must be >= 0")
  }
  if (alpha + beta >= 1) {
    stop("need alpha + beta < 1 for a stationary simulation")
  }

  ret <- numeric(n)
  eps <- numeric(n)
  sigma2 <- numeric(n)

  sigma2[1] <- omega / (1 - alpha - beta)
  eps[1] <- sqrt(sigma2[1]) * rnorm(1)
  ret[1] <- mu + eps[1]

  if (n >= 2) {
    for (i in 2:n) {
      sigma2[i] <- omega + alpha * eps[i - 1]^2 + beta * sigma2[i - 1]
      eps[i] <- sqrt(sigma2[i]) * rnorm(1)
      ret[i] <- mu + eps[i]
    }
  }

  list(ret = ret, eps = eps, sigma2 = sigma2)
}

acf_lags <- function(x, lag_max) {
  x <- x[is.finite(x)]
  n <- length(x)

  if (lag_max >= n) {
    stop("lag_max must be less than length(x)")
  }

  m <- mean(x)
  xc <- x - m
  denom <- sum(xc^2)

  acf_vals <- numeric(lag_max)
  for (k in 1:lag_max) {
    acf_vals[k] <- sum(xc[1:(n - k)] * xc[(k + 1):n]) / denom
  }

  acf_vals
}

summary_stats <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)

  if (n < 2) {
    stop("need at least 2 observations")
  }

  m <- mean(x)
  xc <- x - m
  s <- sqrt(mean(xc^2))

  if (s == 0) {
    skew <- NA_real_
    ex_kurt <- NA_real_
  } else {
    z <- xc / s
    skew <- mean(z^3)
    ex_kurt <- mean(z^4) - 3
  }

  c(mean = m, sd = s, skew = skew, ex_kurt = ex_kurt)
}

garch11_theoretical_stats <- function(omega, alpha, beta, mu = 0) {
  var_ret <- omega / (1 - alpha - beta)
  den4 <- 1 - 3 * alpha^2 - 2 * alpha * beta - beta^2

  if (den4 <= 0) {
    ex_kurt <- NA_real_
  } else {
    kurt <- 3 * (1 - (alpha + beta)^2) / den4
    ex_kurt <- kurt - 3
  }

  c(mean = mu, sd = sqrt(var_ret), skew = 0, ex_kurt = ex_kurt)
}

garch11_theoretical_sq_acf <- function(omega, alpha, beta, mu = 0, lag_max = 10) {
  phi <- alpha + beta
  den4 <- 1 - 3 * alpha^2 - 2 * alpha * beta - beta^2

  if (den4 <= 0) {
    return(rep(NA_real_, lag_max))
  }

  rho1_eps2 <- alpha * (1 - beta * phi) / (1 - 2 * beta * phi + beta^2)
  rho_eps2 <- numeric(lag_max)
  rho_eps2[1] <- rho1_eps2

  if (lag_max >= 2) {
    for (k in 2:lag_max) {
      rho_eps2[k] <- phi^(k - 1) * rho1_eps2
    }
  }

  if (mu == 0) {
    return(rho_eps2)
  }

  var_ret <- omega / (1 - alpha - beta)
  kurt <- 3 * (1 - (alpha + beta)^2) / den4
  var_eps2 <- var_ret^2 * (kurt - 1)

  scale_fac <- var_eps2 / (var_eps2 + 4 * mu^2 * var_ret)

  scale_fac * rho_eps2
}

garch11_filter <- function(ret, omega, alpha, beta, mu = 0) {
  n <- length(ret)

  if (omega <= 0) {
    stop("omega must be > 0")
  }
  if (alpha < 0 || beta < 0 || alpha + beta >= 1) {
    stop("invalid garch parameters")
  }

  eps <- ret - mu
  sigma2 <- numeric(n)

  sigma2[1] <- omega / (1 - alpha - beta)
  if (!is.finite(sigma2[1]) || sigma2[1] <= 0) {
    sigma2[1] <- mean(eps^2)
  }

  if (n >= 2) {
    for (i in 2:n) {
      sigma2[i] <- omega + alpha * eps[i - 1]^2 + beta * sigma2[i - 1]
      if (!is.finite(sigma2[i]) || sigma2[i] <= 0) {
        sigma2[i] <- 1e-12
      }
    }
  }

  z <- eps / sqrt(sigma2)

  list(eps = eps, sigma2 = sigma2, z = z)
}

to_ab <- function(par) {
  s <- 0.999 * plogis(par[1])
  w <- plogis(par[2])

  alpha <- s * w
  beta <- s * (1 - w)

  c(alpha = alpha, beta = beta)
}

fit_garch11_mom <- function(ret, nlags = 5) {
  if (length(ret) <= nlags) {
    stop("need more observations than nlags")
  }

  stats_target <- summary_stats(ret)
  mu_hat <- unname(stats_target["mean"])
  var_hat <- unname(stats_target["sd"]^2)
  ex_kurt_target <- unname(stats_target["ex_kurt"])
  acf_target <- acf_lags(ret^2, nlags)

  objective <- function(par) {
    ab <- to_ab(par)
    alpha <- unname(ab["alpha"])
    beta <- unname(ab["beta"])
    omega <- var_hat * (1 - alpha - beta)

    if (!is.finite(omega) || omega <= 0) {
      return(1e12)
    }

    theo_stats <- garch11_theoretical_stats(omega, alpha, beta, mu_hat)
    theo_acf <- garch11_theoretical_sq_acf(omega, alpha, beta, mu_hat, nlags)

    if (any(!is.finite(theo_stats)) || any(!is.finite(theo_acf))) {
      return(1e12)
    }

    term_acf <- mean((acf_target - theo_acf)^2)
    term_kurt <- ((ex_kurt_target - theo_stats["ex_kurt"]) /
      (1 + abs(ex_kurt_target)))^2

    term <- term_acf + term_kurt

    if (!is.finite(term)) {
      term <- 1e12
    }

    term
  }

  phi0 <- 0.90
  if (nlags >= 2 && is.finite(acf_target[1]) && is.finite(acf_target[2]) &&
      acf_target[1] > 0) {
    phi0 <- min(0.98, max(0.05, acf_target[2] / acf_target[1]))
  }

  s_grid <- sort(unique(c(phi0, 0.70, 0.85, 0.92, 0.97)))
  w_grid <- c(0.05, 0.10, 0.20, 0.30, 0.40)

  best <- NULL

  for (s0 in s_grid) {
    for (w0 in w_grid) {
      p0 <- c(
        qlogis(min(max(s0 / 0.999, 1e-6), 1 - 1e-6)),
        qlogis(min(max(w0, 1e-6), 1 - 1e-6))
      )

      fit <- optim(
        p0,
        objective,
        method = "Nelder-Mead",
        control = list(maxit = 5000)
      )

      if (is.null(best) || fit$value < best$value) {
        best <- fit
      }
    }
  }

  ab <- to_ab(best$par)
  alpha_hat <- unname(ab["alpha"])
  beta_hat <- unname(ab["beta"])
  omega_hat <- var_hat * (1 - alpha_hat - beta_hat)

  filt <- garch11_filter(ret, omega_hat, alpha_hat, beta_hat, mu_hat)

  list(
    mu = mu_hat,
    omega = omega_hat,
    alpha = alpha_hat,
    beta = beta_hat,
    value = best$value,
    stats_target = stats_target,
    acf_target = acf_target,
    stats_fit = garch11_theoretical_stats(omega_hat, alpha_hat, beta_hat, mu_hat),
    acf_fit = garch11_theoretical_sq_acf(omega_hat, alpha_hat, beta_hat, mu_hat, nlags),
    sigma2 = filt$sigma2,
    eps = filt$eps,
    residuals = filt$z
  )
}

jarque_bera_test <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)

  if (n < 2) {
    return(c(jb_stat = NA_real_, jb_p = NA_real_))
  }

  m <- mean(x)
  xc <- x - m
  s <- sqrt(mean(xc^2))

  if (s == 0) {
    return(c(jb_stat = NA_real_, jb_p = NA_real_))
  }

  z <- xc / s
  skew <- mean(z^3)
  kurt <- mean(z^4)

  jb <- n / 6 * (skew^2 + 0.25 * (kurt - 3)^2)
  pval <- 1 - pchisq(jb, df = 2)

  c(jb_stat = jb, jb_p = pval)
}

shapiro_test_safe <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)

  if (n < 3) {
    return(c(sw_w = NA_real_, sw_p = NA_real_))
  }

  if (n > 5000) {
    idx <- unique(round(seq(1, n, length.out = 5000)))
    x <- x[idx]
  }

  out <- shapiro.test(x)

  c(sw_w = unname(out$statistic), sw_p = out$p.value)
}

normality_summary <- function(x) {
  s <- summary_stats(x)
  jb <- jarque_bera_test(x)
  sw <- shapiro_test_safe(x)

  c(
    jb_stat = jb["jb_stat"],
    jb_p = jb["jb_p"],
    sw_w = sw["sw_w"],
    sw_p = sw["sw_p"],
    skew = s["skew"],
    ex_kurt = s["ex_kurt"]
  )
}

fmt_num <- function(x, width = 12, digits = 6) {
  out <- character(length(x))

  for (i in seq_along(x)) {
    if (is.na(x[i])) {
      out[i] <- sprintf(paste0("%", width, "s"), "NA")
    } else {
      out[i] <- sprintf(paste0("%", width, ".", digits, "f"), x[i])
    }
  }

  out
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  n <- if (length(args) >= 1) as.integer(args[1]) else 10000L
  outfile <- if (length(args) >= 2) args[2] else "garch_returns.txt"
  omega <- if (length(args) >= 3) as.numeric(args[3]) else 0.01
  alpha <- if (length(args) >= 4) as.numeric(args[4]) else 0.08
  beta <- if (length(args) >= 5) as.numeric(args[5]) else 0.90
  mu <- if (length(args) >= 6) as.numeric(args[6]) else 0
  seed <- if (length(args) >= 7) as.integer(args[7]) else NULL
  nlags_fit <- if (length(args) >= 8) as.integer(args[8]) else 5L

  sim <- simulate_garch11(
    n = n,
    omega = omega,
    alpha = alpha,
    beta = beta,
    mu = mu,
    seed = seed
  )

  write.table(
    data.frame(ret = sim$ret),
    file = outfile,
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE
  )

  est <- fit_garch11_mom(sim$ret, nlags = nlags_fit)

  sq_acf_sim <- acf_lags(sim$ret^2, 10)
  sq_acf_est <- garch11_theoretical_sq_acf(
    est$omega, est$alpha, est$beta, est$mu, 10
  )
  sq_acf_true <- garch11_theoretical_sq_acf(
    omega, alpha, beta, mu, 10
  )
  sq_acf_diff <- sq_acf_sim - sq_acf_true

  stats_sim <- summary_stats(sim$ret)
  stats_est <- garch11_theoretical_stats(
    est$omega, est$alpha, est$beta, est$mu
  )
  stats_true <- garch11_theoretical_stats(
    omega, alpha, beta, mu
  )
  stats_diff <- stats_sim - stats_true

  param_est <- c(
    mu = est$mu,
    omega = est$omega,
    alpha = est$alpha,
    beta = est$beta
  )
  param_true <- c(
    mu = mu,
    omega = omega,
    alpha = alpha,
    beta = beta
  )
  param_diff <- param_est - param_true

  normal_ret <- normality_summary(sim$ret)
  normal_res <- normality_summary(est$residuals)

  cat("ACF of squared returns\n")
  acf_labels <- c("", as.character(1:10))
  cat(paste(sprintf("%12s", acf_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "simulated"),
      paste(fmt_num(sq_acf_sim), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "estimated"),
      paste(fmt_num(sq_acf_est), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "true"),
      paste(fmt_num(sq_acf_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "diff"),
      paste(fmt_num(sq_acf_diff), collapse = ""),
      "\n", sep = "")

  cat("\n")

  stat_labels <- c("", "mean", "sd", "skew", "ex_kurt")
  cat(paste(sprintf("%12s", stat_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "simulated"),
      paste(fmt_num(stats_sim), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "estimated"),
      paste(fmt_num(stats_est), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "true"),
      paste(fmt_num(stats_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "diff"),
      paste(fmt_num(stats_diff), collapse = ""),
      "\n", sep = "")

  cat("\n")

  param_labels <- c("", "mu", "omega", "alpha", "beta")
  cat("Estimated GARCH(1,1) parameters\n")
  cat(paste(sprintf("%12s", param_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "estimated"),
      paste(fmt_num(param_est), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "true"),
      paste(fmt_num(param_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "diff"),
      paste(fmt_num(param_diff), collapse = ""),
      "\n", sep = "")

  cat("\n")

  norm_labels <- c("", "jb_stat", "jb_p", "sw_w", "sw_p", "skew", "ex_kurt")
  cat("Normality tests\n")
  cat("(residuals are standardized residuals from the estimated GARCH model)\n")
  cat(paste(sprintf("%12s", norm_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "returns"),
      paste(fmt_num(normal_ret), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "residuals"),
      paste(fmt_num(normal_res), collapse = ""),
      "\n", sep = "")

  cat("\n")
  cat("moment fit used mean, sd, ex_kurt, and lags 1 to", nlags_fit,
      "of the acf of squared returns\n")
  cat("objective =", sprintf("%.6f", est$value), "\n")
  cat("wrote", n, "returns to", outfile, "\n")
  cat("omega =", omega, "alpha =", alpha, "beta =", beta,
      "mu =", mu, "seed =", seed, "\n")

  if (is.na(stats_true["ex_kurt"]) || any(is.na(sq_acf_true))) {
    cat("note: true ex_kurt and true squared-return acf require a finite fourth moment\n")
    cat("      condition: 3*alpha^2 + 2*alpha*beta + beta^2 < 1\n")
  }
}

main()
