# xgarch_sim.R
# simulate returns from a garch(1,1) model using base r only
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
  n <- length(x)
  m <- mean(x)
  s <- sd(x)

  if (s == 0) {
    skew <- NA_real_
    ex_kurt <- NA_real_
  } else {
    z <- (x - m) / s
    skew <- mean(z^3)
    ex_kurt <- mean(z^4) - 3
  }

  c(mean = m, sd = s, skew = skew, ex_kurt = ex_kurt)
}

garch11_true_stats <- function(omega, alpha, beta, mu = 0) {
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

garch11_true_sq_acf <- function(omega, alpha, beta, mu = 0, lag_max = 10) {
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

  sq_acf_sim <- acf_lags(sim$ret^2, 10)
  sq_acf_true <- garch11_true_sq_acf(omega, alpha, beta, mu, 10)
  sq_acf_diff <- sq_acf_sim - sq_acf_true

  stats_sim <- summary_stats(sim$ret)
  stats_true <- garch11_true_stats(omega, alpha, beta, mu)
  stats_diff <- stats_sim - stats_true

  cat("ACF of squared returns\n")
  acf_labels <- c("", as.character(1:10))
  cat(paste(sprintf("%12s", acf_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "simulated"),
      paste(fmt_num(sq_acf_sim), collapse = ""),
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
  cat(sprintf("%12s", "true"),
      paste(fmt_num(stats_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "diff"),
      paste(fmt_num(stats_diff), collapse = ""),
      "\n", sep = "")

  cat("\n")
  cat("wrote", n, "returns to", outfile, "\n")
  cat("omega =", omega, "alpha =", alpha, "beta =", beta,
      "mu =", mu, "seed =", seed, "\n")

  if (any(is.na(stats_true["ex_kurt"])) || any(is.na(sq_acf_true))) {
    cat("note: true ex_kurt and true squared-return acf require a finite fourth moment\n")
    cat("      condition: 3*alpha^2 + 2*alpha*beta + beta^2 < 1\n")
  }
}

main()
