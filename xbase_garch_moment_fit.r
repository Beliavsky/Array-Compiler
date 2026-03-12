timing_enabled <- TRUE

# xgarch_sim.r
# simulate returns from a garch(1,1) model using base r only
# write the simulated returns to a file
# print:
#   1) acf of squared returns
#   2) moments of returns
#   3) direct estimates of garch(1,1) parameters with no optimizer
#   4) normality tests for returns and standardized residuals
#   5) elapsed times by section if timing is enabled
#
# model:
#   r_t = mu + eps_t
#   eps_t = sqrt(sigma2_t) * z_t,   z_t ~ n(0,1)
#   sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
#
# command line arguments:
#   1: n
#   2: outfile
#   3: omega
#   4: alpha
#   5: beta
#   6: mu
#   7: seed

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

garch11_uncond_sd <- function(omega, alpha, beta) {
  sqrt(omega / (1 - alpha - beta))
}

garch11_theoretical_stats <- function(omega, alpha, beta, mu = 0) {
  var_ret <- omega / (1 - alpha - beta)
  den4 <- 1 - 3 * alpha^2 - 2 * alpha * beta - beta^2

  if (den4 <= 0) {
    ex_kurt <- NA_real_
  } else {
    ex_kurt <- 6 * alpha^2 / den4
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
  ex_kurt <- 6 * alpha^2 / den4
  kurt <- ex_kurt + 3
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

clip01 <- function(x, lo = 1e-6, hi = 0.999) {
  min(max(x, lo), hi)
}

estimate_phi_from_sq_acf <- function(acf_sq, nratios = 4) {
  m <- min(nratios, length(acf_sq) - 1)

  if (m < 1) {
    return(0.90)
  }

  ratios <- numeric(0)
  for (i in 1:m) {
    if (is.finite(acf_sq[i]) &&
        is.finite(acf_sq[i + 1]) &&
        acf_sq[i] > 0 &&
        acf_sq[i + 1] > 0) {
      ratios <- c(ratios, acf_sq[i + 1] / acf_sq[i])
    }
  }

  ratios <- ratios[is.finite(ratios) & ratios > 0 & ratios < 0.999]

  if (length(ratios) == 0) {
    return(0.90)
  }

  clip01(median(ratios), lo = 0.01, hi = 0.999)
}

estimate_alpha_from_phi_kurt <- function(phi, ex_kurt) {
  c0 <- 1 - phi^2

  if (!is.finite(phi) || !is.finite(ex_kurt) || c0 <= 0 || ex_kurt <= 0) {
    return(NA_real_)
  }

  alpha2 <- ex_kurt * c0 / (2 * (ex_kurt + 3))

  if (!is.finite(alpha2) || alpha2 <= 0) {
    return(NA_real_)
  }

  alpha <- sqrt(alpha2)

  if (alpha <= 0 || alpha >= phi) {
    return(NA_real_)
  }

  alpha
}

estimate_alpha_from_phi_rho1 <- function(phi, rho1) {
  c0 <- 1 - phi^2

  if (!is.finite(phi) || !is.finite(rho1) || c0 <= 0 || rho1 <= 0) {
    return(NA_real_)
  }

  acoef <- rho1 - phi
  bcoef <- -c0
  ccoef <- rho1 * c0

  if (abs(acoef) < 1e-12) {
    alpha <- rho1
    if (alpha > 0 && alpha < phi) {
      return(alpha)
    }
    return(NA_real_)
  }

  disc <- bcoef^2 - 4 * acoef * ccoef
  if (!is.finite(disc) || disc < 0) {
    return(NA_real_)
  }

  roots <- c(
    (-bcoef + sqrt(disc)) / (2 * acoef),
    (-bcoef - sqrt(disc)) / (2 * acoef)
  )

  roots <- roots[is.finite(roots) & roots > 0 & roots < phi]

  if (length(roots) == 0) {
    return(NA_real_)
  }

  roots[1]
}

fit_garch11_direct <- function(ret) {
  stats_target <- summary_stats(ret)
  mu_hat <- unname(stats_target["mean"])
  xc <- ret - mu_hat
  var_hat <- mean(xc^2)
  ex_kurt_hat <- unname(stats_target["ex_kurt"])

  acf_sq_centered <- acf_lags(xc^2, 10)
  rho1_hat <- acf_sq_centered[1]
  phi_hat <- estimate_phi_from_sq_acf(acf_sq_centered, nratios = 4)

  alpha_kurt <- estimate_alpha_from_phi_kurt(phi_hat, ex_kurt_hat)
  alpha_rho1 <- estimate_alpha_from_phi_rho1(phi_hat, rho1_hat)

  if (is.finite(alpha_kurt)) {
    alpha_hat <- alpha_kurt
  } else if (is.finite(alpha_rho1)) {
    alpha_hat <- alpha_rho1
  } else {
    alpha_hat <- min(0.10, 0.25 * phi_hat)
  }

  beta_hat <- phi_hat - alpha_hat

  alpha_hat <- clip01(alpha_hat, lo = 1e-6, hi = phi_hat - 1e-6)
  beta_hat <- clip01(beta_hat, lo = 1e-6, hi = 0.999 - alpha_hat)

  if (alpha_hat + beta_hat >= 0.999) {
    beta_hat <- 0.999 - alpha_hat
  }

  omega_hat <- var_hat * (1 - alpha_hat - beta_hat)
  omega_hat <- max(omega_hat, 1e-12)

  filt <- garch11_filter(ret, omega_hat, alpha_hat, beta_hat, mu_hat)

  list(
    mu = mu_hat,
    omega = omega_hat,
    alpha = alpha_hat,
    beta = beta_hat,
    sd_uncond = garch11_uncond_sd(omega_hat, alpha_hat, beta_hat),
    stats_target = stats_target,
    acf_sq_centered = acf_sq_centered,
    rho1_hat = rho1_hat,
    phi_hat = phi_hat,
    alpha_kurt = alpha_kurt,
    alpha_rho1 = alpha_rho1,
    stats_fit = garch11_theoretical_stats(omega_hat, alpha_hat, beta_hat, mu_hat),
    acf_fit = garch11_theoretical_sq_acf(omega_hat, alpha_hat, beta_hat, mu_hat, 10),
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
      out[i] <- sprintf(paste0("%", width, "s"), "na")
    } else {
      out[i] <- sprintf(paste0("%", width, ".", digits, "f"), x[i])
    }
  }

  out
}

timer_new <- function(enabled = TRUE) {
  list(enabled = enabled, elapsed = numeric(0))
}

timer_add <- function(timer, name, seconds) {
  if (timer$enabled) {
    timer$elapsed[name] <- seconds
  }
  timer
}

timer_print <- function(timer) {
  if (!timer$enabled) {
    return(invisible(NULL))
  }

  vals <- unname(timer$elapsed)
  nms <- names(timer$elapsed)

  cat("elapsed times in seconds\n")
  cat(sprintf("%20s %12s\n", "section", "elapsed"))
  for (i in seq_along(vals)) {
    cat(sprintf("%20s %12.6f\n", nms[i], vals[i]))
  }
  cat(sprintf("%20s %12.6f\n", "total", sum(vals)))
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  n <- if (length(args) >= 1) as.integer(args[1]) else 1000000L
  outfile <- if (length(args) >= 2) args[2] else "garch_returns.txt"
  omega <- if (length(args) >= 3) as.numeric(args[3]) else 0.01
  alpha <- if (length(args) >= 4) as.numeric(args[4]) else 0.08
  beta <- if (length(args) >= 5) as.numeric(args[5]) else 0.90
  mu <- if (length(args) >= 6) as.numeric(args[6]) else 0
  seed <- if (length(args) >= 7) as.integer(args[7]) else NULL

  timer <- timer_new(timing_enabled)

  t0 <- proc.time()[["elapsed"]]
  sim <- simulate_garch11(
    n = n,
    omega = omega,
    alpha = alpha,
    beta = beta,
    mu = mu,
    seed = seed
  )
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "simulate", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
  write.table(
    data.frame(ret = sim$ret),
    file = outfile,
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE
  )
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "write_file", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
  est <- fit_garch11_direct(sim$ret)
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "estimate", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
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
    beta = est$beta,
    sd_uncond = est$sd_uncond
  )
  param_true <- c(
    mu = mu,
    omega = omega,
    alpha = alpha,
    beta = beta,
    sd_uncond = garch11_uncond_sd(omega, alpha, beta)
  )
  param_diff <- param_est - param_true

  normal_ret <- normality_summary(sim$ret)
  normal_res <- normality_summary(est$residuals)
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "diagnostics", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
  cat("acf of squared returns\n")
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

  param_labels <- c("", "mu", "omega", "alpha", "beta", "sd_uncond")
  cat("estimated garch(1,1) parameters\n")
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
  cat("normality tests\n")
  cat("(residuals are standardized residuals from the direct garch estimate)\n")
  cat(paste(sprintf("%12s", norm_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "returns"),
      paste(fmt_num(normal_ret), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "residuals"),
      paste(fmt_num(normal_res), collapse = ""),
      "\n", sep = "")

  cat("\n")
  cat("direct estimator details\n")
  detail_labels <- c("", "rho1_sq", "phi", "alpha_kurt", "alpha_rho1")
  detail_vals <- c(
    est$rho1_hat,
    est$phi_hat,
    est$alpha_kurt,
    est$alpha_rho1
  )
  cat(paste(sprintf("%12s", detail_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "estimate"),
      paste(fmt_num(detail_vals), collapse = ""),
      "\n", sep = "")

  cat("\n")
  cat("wrote", n, "returns to", outfile, "\n")
  cat("omega =", omega, "alpha =", alpha, "beta =", beta,
      "mu =", mu, "seed =", seed, "\n")

  if (is.na(stats_true["ex_kurt"]) || any(is.na(sq_acf_true))) {
    cat("note: true ex_kurt and true squared-return acf require a finite fourth moment\n")
    cat("      condition: 3*alpha^2 + 2*alpha*beta + beta^2 < 1\n")
  }
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "print_main_output", t1 - t0)

  cat("\n")
  timer_print(timer)
}

main()
