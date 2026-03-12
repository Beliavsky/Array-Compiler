timing_enabled <- TRUE

n <- 2000L
outfile <- "ccc_garch_returns.txt"
seed <- NULL
burn_in <- 1000L

mc_corr_n <- 50000L
mc_corr_burn <- 5000L

omega_true <- c(0.010, 0.015, 0.020)
alpha_true <- c(0.080, 0.100, 0.070)
beta_true <- c(0.900, 0.850, 0.880)
mu_true <- c(0.000, 0.000, 0.000)

R_true <- matrix(
  c(
    1.00, 0.45, 0.25,
    0.45, 1.00, 0.35,
    0.25, 0.35, 1.00
  ),
  nrow = 3,
  byrow = TRUE
)

make_spd_corr <- function(R, eps = 1e-8) {
  R <- 0.5 * (R + t(R))
  eig <- eigen(R, symmetric = TRUE)
  vals <- pmax(eig$values, eps)
  X <- eig$vectors %*% diag(vals, nrow = length(vals)) %*% t(eig$vectors)
  d <- sqrt(diag(X))
  X <- diag(1 / d, nrow = length(d)) %*% X %*% diag(1 / d, nrow = length(d))
  X <- 0.5 * (X + t(X))
  diag(X) <- 1
  X
}

validate_ccc_inputs <- function(omega, alpha, beta, mu, R) {
  p <- length(omega)

  if (length(alpha) != p || length(beta) != p || length(mu) != p) {
    stop("omega, alpha, beta, and mu must have the same length")
  }
  if (!is.matrix(R) || nrow(R) != p || ncol(R) != p) {
    stop("R_true must be a square matrix with dimension length(omega_true)")
  }
  if (any(omega <= 0)) {
    stop("all omega values must be > 0")
  }
  if (any(alpha < 0) || any(beta < 0)) {
    stop("all alpha and beta values must be >= 0")
  }
  if (any(alpha + beta >= 1)) {
    stop("need alpha + beta < 1 for each asset")
  }
}

simulate_ccc_garch <- function(n, omega, alpha, beta, mu, R, seed = NULL, burn_in = 0L) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  validate_ccc_inputs(omega, alpha, beta, mu, R)

  p <- length(omega)
  R <- make_spd_corr(R)
  total_n <- n + burn_in

  asset_names <- if (!is.null(names(omega))) {
    names(omega)
  } else {
    paste0("asset", seq_len(p))
  }

  z0 <- matrix(rnorm(total_n * p), nrow = total_n, ncol = p)
  C <- chol(R)
  z <- z0 %*% C

  sigma2 <- matrix(0, nrow = total_n, ncol = p)
  eps <- matrix(0, nrow = total_n, ncol = p)
  ret <- matrix(0, nrow = total_n, ncol = p)

  sigma2[1, ] <- omega / (1 - alpha - beta)
  eps[1, ] <- sqrt(sigma2[1, ]) * z[1, ]
  ret[1, ] <- mu + eps[1, ]

  if (total_n >= 2) {
    for (t in 2:total_n) {
      sigma2[t, ] <- omega + alpha * eps[t - 1, ]^2 + beta * sigma2[t - 1, ]
      eps[t, ] <- sqrt(sigma2[t, ]) * z[t, ]
      ret[t, ] <- mu + eps[t, ]
    }
  }

  if (burn_in > 0) {
    keep <- (burn_in + 1):total_n
    sigma2 <- sigma2[keep, , drop = FALSE]
    eps <- eps[keep, , drop = FALSE]
    ret <- ret[keep, , drop = FALSE]
    z <- z[keep, , drop = FALSE]
  }

  colnames(sigma2) <- asset_names
  colnames(eps) <- asset_names
  colnames(ret) <- asset_names
  colnames(z) <- asset_names

  list(ret = ret, eps = eps, sigma2 = sigma2, z = z)
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

fit_ccc_garch_direct <- function(ret_mat) {
  p <- ncol(ret_mat)
  n <- nrow(ret_mat)

  fits <- vector("list", p)
  mu_fit <- numeric(p)
  omega_fit <- numeric(p)
  alpha_fit <- numeric(p)
  beta_fit <- numeric(p)
  sd_uncond_fit <- numeric(p)
  z_fit <- matrix(NA_real_, nrow = n, ncol = p)

  for (i in seq_len(p)) {
    fits[[i]] <- fit_garch11_direct(ret_mat[, i])
    mu_fit[i] <- fits[[i]]$mu
    omega_fit[i] <- fits[[i]]$omega
    alpha_fit[i] <- fits[[i]]$alpha
    beta_fit[i] <- fits[[i]]$beta
    sd_uncond_fit[i] <- fits[[i]]$sd_uncond
    z_fit[, i] <- fits[[i]]$residuals
  }

  colnames(z_fit) <- colnames(ret_mat)
  R_fit <- cor(z_fit)
  R_fit <- make_spd_corr(R_fit)
  dimnames(R_fit) <- list(colnames(ret_mat), colnames(ret_mat))

  names(mu_fit) <- colnames(ret_mat)
  names(omega_fit) <- colnames(ret_mat)
  names(alpha_fit) <- colnames(ret_mat)
  names(beta_fit) <- colnames(ret_mat)
  names(sd_uncond_fit) <- colnames(ret_mat)

  list(
    fits = fits,
    mu_fit = mu_fit,
    omega_fit = omega_fit,
    alpha_fit = alpha_fit,
    beta_fit = beta_fit,
    sd_uncond_fit = sd_uncond_fit,
    z_fit = z_fit,
    R_fit = R_fit
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

print_matrix_block <- function(title, mat) {
  cat(title, "\n", sep = "")
  cn <- colnames(mat)
  rn <- rownames(mat)
  cat(sprintf("%12s", ""), paste(sprintf("%12s", cn), collapse = ""), "\n", sep = "")
  for (i in seq_len(nrow(mat))) {
    cat(sprintf("%12s", rn[i]), paste(fmt_num(mat[i, ]), collapse = ""), "\n", sep = "")
  }
  cat("\n")
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

asset_report <- function(asset_name, ret, fit, omega_true_i, alpha_true_i, beta_true_i, mu_true_i) {
  sq_acf_sim <- acf_lags(ret^2, 10)
  sq_acf_fit <- garch11_theoretical_sq_acf(
    fit$omega, fit$alpha, fit$beta, fit$mu, 10
  )
  sq_acf_true <- garch11_theoretical_sq_acf(
    omega_true_i, alpha_true_i, beta_true_i, mu_true_i, 10
  )
  sq_acf_sim_true <- sq_acf_sim - sq_acf_true
  sq_acf_fit_true <- sq_acf_fit - sq_acf_true

  stats_sim <- summary_stats(ret)
  stats_fit <- garch11_theoretical_stats(
    fit$omega, fit$alpha, fit$beta, fit$mu
  )
  stats_true <- garch11_theoretical_stats(
    omega_true_i, alpha_true_i, beta_true_i, mu_true_i
  )
  stats_sim_true <- stats_sim - stats_true
  stats_fit_true <- stats_fit - stats_true

  param_fit <- c(
    mu = fit$mu,
    omega = fit$omega,
    alpha = fit$alpha,
    beta = fit$beta,
    sd_uncond = fit$sd_uncond
  )
  param_true <- c(
    mu = mu_true_i,
    omega = omega_true_i,
    alpha = alpha_true_i,
    beta = beta_true_i,
    sd_uncond = garch11_uncond_sd(omega_true_i, alpha_true_i, beta_true_i)
  )
  param_fit_true <- param_fit - param_true

  normal_ret <- normality_summary(ret)
  normal_res <- normality_summary(fit$residuals)

  cat("============================================================\n")
  cat(asset_name, "\n")
  cat("============================================================\n")

  cat("acf of squared returns\n")
  acf_labels <- c("", as.character(1:10))
  cat(paste(sprintf("%12s", acf_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "simulated"),
      paste(fmt_num(sq_acf_sim), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "fit"),
      paste(fmt_num(sq_acf_fit), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "true"),
      paste(fmt_num(sq_acf_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "sim-true"),
      paste(fmt_num(sq_acf_sim_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "fit-true"),
      paste(fmt_num(sq_acf_fit_true), collapse = ""),
      "\n", sep = "")

  cat("\n")

  stat_labels <- c("", "mean", "sd", "skew", "ex_kurt")
  cat(paste(sprintf("%12s", stat_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "simulated"),
      paste(fmt_num(stats_sim), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "fit"),
      paste(fmt_num(stats_fit), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "true"),
      paste(fmt_num(stats_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "sim-true"),
      paste(fmt_num(stats_sim_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "fit-true"),
      paste(fmt_num(stats_fit_true), collapse = ""),
      "\n", sep = "")

  cat("\n")

  param_labels <- c("", "mu", "omega", "alpha", "beta", "sd_uncond")
  cat("estimated garch(1,1) parameters\n")
  cat(paste(sprintf("%12s", param_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "fit"),
      paste(fmt_num(param_fit), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "true"),
      paste(fmt_num(param_true), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "fit-true"),
      paste(fmt_num(param_fit_true), collapse = ""),
      "\n", sep = "")

  cat("\n")

  norm_labels <- c("", "jb_stat", "jb_p", "sw_w", "sw_p", "skew", "ex_kurt")
  cat("normality tests\n")
  cat("(residuals are standardized residuals from the direct garch fit)\n")
  cat(paste(sprintf("%12s", norm_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "returns"),
      paste(fmt_num(normal_ret), collapse = ""),
      "\n", sep = "")
  cat(sprintf("%12s", "residuals"),
      paste(fmt_num(normal_res), collapse = ""),
      "\n", sep = "")

  cat("\n")

  detail_labels <- c("", "rho1_sq", "phi", "alpha_kurt", "alpha_rho1")
  detail_vals <- c(
    fit$rho1_hat,
    fit$phi_hat,
    fit$alpha_kurt,
    fit$alpha_rho1
  )
  cat("direct estimator details\n")
  cat(paste(sprintf("%12s", detail_labels), collapse = ""), "\n", sep = "")
  cat(sprintf("%12s", "estimate"),
      paste(fmt_num(detail_vals), collapse = ""),
      "\n", sep = "")

  cat("\n")
}

main <- function() {
  validate_ccc_inputs(omega_true, alpha_true, beta_true, mu_true, R_true)

  p <- length(omega_true)
  asset_names <- if (!is.null(names(omega_true))) {
    names(omega_true)
  } else {
    paste0("asset", seq_len(p))
  }

  names(omega_true) <<- asset_names
  names(alpha_true) <<- asset_names
  names(beta_true) <<- asset_names
  names(mu_true) <<- asset_names
  dimnames(R_true) <<- list(asset_names, asset_names)

  timer <- timer_new(timing_enabled)

  t0 <- proc.time()[["elapsed"]]
  sim <- simulate_ccc_garch(
    n = n,
    omega = omega_true,
    alpha = alpha_true,
    beta = beta_true,
    mu = mu_true,
    R = R_true,
    seed = seed,
    burn_in = burn_in
  )
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "simulate", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
  write.table(
    data.frame(sim$ret),
    file = outfile,
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE
  )
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "write_file", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
  fit <- fit_ccc_garch_direct(sim$ret)
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "fit_univariate_ccc", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
  corr_ret_emp <- cor(sim$ret)
  corr_z_emp <- cor(sim$z)
  corr_z_fit <- fit$R_fit
  corr_z_true <- R_true

  sim_fit_mc <- simulate_ccc_garch(
    n = mc_corr_n,
    omega = fit$omega_fit,
    alpha = fit$alpha_fit,
    beta = fit$beta_fit,
    mu = fit$mu_fit,
    R = fit$R_fit,
    seed = if (is.null(seed)) NULL else seed + 1000L,
    burn_in = mc_corr_burn
  )
  corr_ret_fit <- cor(sim_fit_mc$ret)

  sim_true_mc <- simulate_ccc_garch(
    n = mc_corr_n,
    omega = omega_true,
    alpha = alpha_true,
    beta = beta_true,
    mu = mu_true,
    R = R_true,
    seed = if (is.null(seed)) NULL else seed + 2000L,
    burn_in = mc_corr_burn
  )
  corr_ret_true <- cor(sim_true_mc$ret)
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "correlation_matrices", t1 - t0)

  t0 <- proc.time()[["elapsed"]]
  for (i in seq_len(p)) {
    asset_report(
      asset_name = asset_names[i],
      ret = sim$ret[, i],
      fit = fit$fits[[i]],
      omega_true_i = omega_true[i],
      alpha_true_i = alpha_true[i],
      beta_true_i = beta_true[i],
      mu_true_i = mu_true[i]
    )
  }

  cat("============================================================\n")
  cat("correlation matrix of returns\n")
  cat("empirical = sample correlation of simulated returns\n")
  cat("fit = monte carlo approximation under fitted ccc-garch model\n")
  cat("true = monte carlo approximation under true ccc-garch model\n")
  cat("============================================================\n\n")

  print_matrix_block("empirical", corr_ret_emp)
  print_matrix_block("fit", corr_ret_fit)
  print_matrix_block("true", corr_ret_true)

  cat("============================================================\n")
  cat("correlation matrix of standardized residuals\n")
  cat("empirical = sample correlation of true simulated shocks z_t\n")
  cat("fit = sample correlation of fitted standardized residuals\n")
  cat("true = input ccc correlation matrix\n")
  cat("============================================================\n\n")

  print_matrix_block("empirical", corr_z_emp)
  print_matrix_block("fit", corr_z_fit)
  print_matrix_block("true", corr_z_true)

  cat("wrote", n, "rows of", p, "-asset returns to", outfile, "\n")
  cat("burn_in =", burn_in, "\n")
  cat("mc_corr_n =", mc_corr_n, "mc_corr_burn =", mc_corr_burn, "\n")

  exk_true <- mapply(
    function(o, a, b, m) garch11_theoretical_stats(o, a, b, m)["ex_kurt"],
    omega_true, alpha_true, beta_true, mu_true
  )

  if (any(is.na(exk_true))) {
    cat("note: some true excess kurtosis values are na because the fourth moment is not finite\n")
  }

  cat("\n")
  t1 <- proc.time()[["elapsed"]]
  timer <- timer_add(timer, "print_output", t1 - t0)

  timer_print(timer)
}

main()
