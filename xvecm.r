# vecm_johansen_base_r.R
#
# Base-R script to:
#   1) simulate a VECM for n > 2 series
#   2) estimate it by the Johansen reduced-rank method
#   3) print estimated - true parameter differences
#
# Model:
#   d y_t = alpha %*% t(beta) %*% y_{t-1}
#           + sum_{i=1}^{p-1} Gamma_i %*% d y_{t-i}
#           + u_t
#
# where:
#   y_t      is n x 1
#   alpha    is n x r
#   beta     is n x r
#   Gamma_i  is n x n
#   u_t ~ N(0, Sigma_u)
#
# Notes:
# - This script uses only base R.
# - The Johansen estimator here uses the known cointegration rank r.
#   That is the natural choice when you simulate from a model and want
#   to compare estimates to the true parameters.
# - beta is normalized so that its first r rows equal I_r. This removes
#   the usual scale / sign / ordering indeterminacy of cointegrating vectors.

rm(list = ls())

set.seed(12345)

# ------------------------------------------------------------
# Simulate a VECM
# ------------------------------------------------------------
simulate_vecm <- function(T, alpha, beta, Gamma_list, sigma_u,
                          burn = 300, y_init = NULL) {
  # T         : number of observations returned after burn-in
  # alpha     : n x r loading matrix
  # beta      : n x r cointegration matrix
  # Gamma_list: list of n x n short-run coefficient matrices
  # sigma_u   : n x n covariance matrix of shocks
  # burn      : burn-in length
  # y_init    : optional initial n-vector for the level series

  n <- nrow(alpha)
  r <- ncol(alpha)
  p <- length(Gamma_list) + 1

  if (nrow(beta) != n || ncol(beta) != r) {
    stop("beta must have dimension n x r.")
  }

  if (!all(dim(sigma_u) == c(n, n))) {
    stop("sigma_u must be n x n.")
  }

  if (length(Gamma_list) > 0) {
    for (i in seq_along(Gamma_list)) {
      if (!all(dim(Gamma_list[[i]]) == c(n, n))) {
        stop("Each Gamma_i must be n x n.")
      }
    }
  }

  # Long-run matrix Pi = alpha %*% t(beta)
  Pi <- alpha %*% t(beta)

  total_T <- T + burn
  if (total_T <= p) {
    stop("T + burn must be larger than p.")
  }

  # Storage for levels y_t
  y <- matrix(0, nrow = total_T, ncol = n)

  # Set initial values for the first p observations
  if (!is.null(y_init)) {
    if (length(y_init) != n) {
      stop("y_init must have length n.")
    }
    for (t in 1:p) {
      y[t, ] <- y_init
    }
  }

  # Cholesky factor for correlated Gaussian shocks
  # chol(sigma_u) in R returns upper-triangular C with t(C) %*% C = sigma_u
  chol_u <- chol(sigma_u)

  # Simulate recursively
  for (t in (p + 1):total_T) {
    # Start with the error-correction term Pi %*% y_{t-1}
    dy_t <- drop(Pi %*% y[t - 1, ])

    # Add lagged differences
    if (p > 1) {
      for (i in 1:(p - 1)) {
        dy_lag_i <- y[t - i, ] - y[t - i - 1, ]
        dy_t <- dy_t + drop(Gamma_list[[i]] %*% dy_lag_i)
      }
    }

    # Add innovation u_t
    u_t <- drop(t(chol_u) %*% rnorm(n))
    dy_t <- dy_t + u_t

    # Convert difference back to level
    y[t, ] <- y[t - 1, ] + dy_t
  }

  # Drop burn-in
  y[(burn + 1):total_T, , drop = FALSE]
}

# ------------------------------------------------------------
# Build the regression matrices used by the Johansen estimator
# ------------------------------------------------------------
make_vecm_matrices <- function(Y, p) {
  # Y: T x n matrix of levels
  # p: VAR lag order in levels, so the VECM has p-1 lagged differences

  Tobs <- nrow(Y)
  n <- ncol(Y)

  if (Tobs <= p) {
    stop("Need more rows in Y than p.")
  }

  # First differences: row k is d y_{k+1}
  dY_all <- diff(Y)

  # Current dependent variable d y_t for t = p+1, ..., Tobs
  # In 1-based indexing this is dY_all rows p:(Tobs-1)
  dY <- dY_all[p:(Tobs - 1), , drop = FALSE]

  # Lagged levels y_{t-1} for t = p+1, ..., Tobs
  Y_lag1 <- Y[p:(Tobs - 1), , drop = FALSE]

  # Stack the lagged differences:
  # W_t = [d y_{t-1}, d y_{t-2}, ..., d y_{t-p+1}]
  if (p == 1) {
    W <- NULL
  } else {
    W_blocks <- vector("list", p - 1)
    for (i in 1:(p - 1)) {
      W_blocks[[i]] <- dY_all[(p - i):(Tobs - 1 - i), , drop = FALSE]
    }
    W <- do.call(cbind, W_blocks)
  }

  list(
    dY = dY,
    Y_lag1 = Y_lag1,
    W = W,
    n_eff = nrow(dY),
    n = n
  )
}

# ------------------------------------------------------------
# Residualize a matrix A on a regressor matrix W
# ------------------------------------------------------------
residualize_on_W <- function(A, W) {
  # Returns residuals from regressing each column of A on W
  # using least squares, all in base R.

  if (is.null(W) || ncol(W) == 0) {
    return(A)
  }

  coef_A_on_W <- qr.solve(W, A)
  A - W %*% coef_A_on_W
}

# ------------------------------------------------------------
# Johansen estimation, base-R implementation
# ------------------------------------------------------------
johansen_fit <- function(Y, p, r, norm_rows = 1:r) {
  # Y         : T x n matrix of levels
  # p         : VAR lag order in levels
  # r         : cointegration rank
  # norm_rows : which rows of beta are forced to equal I_r
  #
  # Returns:
  #   beta_hat, alpha_hat, Gamma_hat, Pi_hat, sigma_u_hat,
  #   Johansen eigenvalues, trace stats, max-eigenvalue stats

  mats <- make_vecm_matrices(Y, p)

  dY <- mats$dY
  Y_lag1 <- mats$Y_lag1
  W <- mats$W
  T_eff <- mats$n_eff
  n <- mats$n

  if (r < 0 || r > n) {
    stop("r must satisfy 0 <= r <= n.")
  }

  if (length(norm_rows) != r) {
    stop("norm_rows must have length r.")
  }

  # Step 1: partial out W from dY and Y_{t-1}
  R0 <- residualize_on_W(dY, W)
  R1 <- residualize_on_W(Y_lag1, W)

  # Step 2: sample moment matrices
  S00 <- crossprod(R0) / T_eff
  S01 <- crossprod(R0, R1) / T_eff
  S10 <- t(S01)
  S11 <- crossprod(R1) / T_eff

  # Step 3: solve the Johansen eigenvalue problem
  #
  # We compute eigenvalues of:
  #   solve(S11) %*% S10 %*% solve(S00) %*% S01
  #
  # The r largest eigenvectors span the estimated cointegration space.
  M <- solve(S11, S10 %*% solve(S00, S01))
  ee <- eigen(M)

  lambda <- Re(ee$values)
  V <- Re(ee$vectors)

  # Sort from largest to smallest
  ord <- order(lambda, decreasing = TRUE)
  lambda <- lambda[ord]
  V <- V[, ord, drop = FALSE]

  # Numerical cleanup: the canonical correlations should lie in [0, 1]
  lambda <- pmin(pmax(lambda, 0), 1)

  # Keep the first r eigenvectors
  beta_raw <- V[, 1:r, drop = FALSE]

  # Normalize beta so that beta[norm_rows, ] = I_r.
  #
  # This is important because beta is only identified up to a nonsingular
  # r x r transformation. The normalization makes comparison to the true
  # beta meaningful.
  beta_top <- beta_raw[norm_rows, , drop = FALSE]
  beta_hat <- beta_raw %*% qr.solve(beta_top, diag(r))

  # Step 4: given beta_hat, estimate alpha and Gamma by OLS
  #
  # Regress dY on:
  #   ECT_t = beta_hat' y_{t-1}
  #   W_t   = lagged differences
  #
  # In row form:
  #   dY = X B + U
  # where:
  #   X = [Y_lag1 %*% beta_hat, W]
  ect <- Y_lag1 %*% beta_hat

  if (is.null(W)) {
    X <- ect
  } else {
    X <- cbind(ect, W)
  }

  B <- qr.solve(X, dY)

  # The first r rows of B correspond to t(alpha)
  alpha_hat <- t(B[1:r, , drop = FALSE])

  # Remaining coefficient blocks are the short-run Gamma_i matrices
  Gamma_hat <- list()
  if (p > 1) {
    col_start <- r + 1
    for (i in 1:(p - 1)) {
      block_i <- B[col_start:(col_start + n - 1), , drop = FALSE]
      Gamma_hat[[i]] <- t(block_i)
      col_start <- col_start + n
    }
  }

  # Residual covariance estimate
  U_hat <- dY - X %*% B
  sigma_u_hat <- crossprod(U_hat) / nrow(U_hat)

  # Long-run matrix estimate
  Pi_hat <- alpha_hat %*% t(beta_hat)

  # Johansen rank-test statistics, printed for information.
  # This script does not use built-in critical values.
  trace_stats <- sapply(0:(n - 1), function(r0) {
    -T_eff * sum(log(1 - lambda[(r0 + 1):n]))
  })

  maxeig_stats <- -T_eff * log(1 - lambda)

  list(
    alpha = alpha_hat,
    beta = beta_hat,
    Gamma = Gamma_hat,
    Pi = Pi_hat,
    sigma_u = sigma_u_hat,
    lambda = lambda,
    trace_stats = trace_stats,
    maxeig_stats = maxeig_stats,
    T_eff = T_eff
  )
}

# ------------------------------------------------------------
# Printing helpers
# ------------------------------------------------------------
print_diff_matrix <- function(name, est, truth, digits = 4) {
  cat("\n", name, ": estimated - true\n", sep = "")
  print(round(est - truth, digits))
}

print_fit_vs_truth <- function(fit, truth, digits = 4) {
  cat("\n========================================\n")
  cat("Johansen eigenvalues\n")
  cat("========================================\n")
  print(round(fit$lambda, digits))

  cat("\n========================================\n")
  cat("Johansen trace statistics\n")
  cat("trace statistic for H0: rank <= r0, r0 = 0, 1, ..., n-1\n")
  cat("========================================\n")
  print(round(fit$trace_stats, digits))

  cat("\n========================================\n")
  cat("Johansen max-eigenvalue statistics\n")
  cat("========================================\n")
  print(round(fit$maxeig_stats, digits))

  cat("\n========================================\n")
  cat("Parameter differences\n")
  cat("========================================\n")

  print_diff_matrix("alpha", fit$alpha, truth$alpha, digits)
  print_diff_matrix("beta", fit$beta, truth$beta, digits)
  print_diff_matrix("Pi", fit$Pi, truth$Pi, digits)

  if (length(truth$Gamma) > 0) {
    for (i in seq_along(truth$Gamma)) {
      print_diff_matrix(
        paste0("Gamma_", i),
        fit$Gamma[[i]],
        truth$Gamma[[i]],
        digits
      )
    }
  }

  print_diff_matrix("Sigma_u", fit$sigma_u, truth$sigma_u, digits)
}

# ------------------------------------------------------------
# Example: n = 4 series, rank r = 2, lag order p = 2
# ------------------------------------------------------------

# Number of series, cointegration rank, and lag order
n <- 4
r <- 2
p <- 2

# True beta, normalized so that the first r rows equal I_r
# This normalization makes beta directly comparable to beta_hat.
beta_true <- rbind(
  diag(r),
  matrix(c(
    -1.20,  0.35,
     0.80, -0.90
  ), nrow = n - r, byrow = TRUE)
)

# True alpha
alpha_true <- matrix(c(
  -0.25,  0.05,
   0.12, -0.18,
   0.06,  0.09,
  -0.04,  0.14
), nrow = n, byrow = TRUE)

# One short-run matrix because p = 2
Gamma_1_true <- matrix(c(
   0.20, 0.02, 0.00, 0.00,
   0.01, 0.15, 0.01, 0.00,
   0.00, 0.02, 0.10, 0.01,
   0.00, 0.00, 0.03, 0.12
), nrow = n, byrow = TRUE)

Gamma_true <- list(Gamma_1_true)

# Innovation covariance matrix
sigma_u_true <- matrix(c(
  1.00, 0.30, 0.20, 0.10,
  0.30, 1.20, 0.25, 0.15,
  0.20, 0.25, 0.90, 0.35,
  0.10, 0.15, 0.35, 1.10
), nrow = n, byrow = TRUE)

# True long-run matrix
Pi_true <- alpha_true %*% t(beta_true)

truth <- list(
  alpha = alpha_true,
  beta = beta_true,
  Gamma = Gamma_true,
  Pi = Pi_true,
  sigma_u = sigma_u_true
)

# Simulate data
T <- 1500
Y <- simulate_vecm(
  T = T,
  alpha = alpha_true,
  beta = beta_true,
  Gamma_list = Gamma_true,
  sigma_u = sigma_u_true,
  burn = 300
)

# Estimate with the Johansen method
fit <- johansen_fit(
  Y = Y,
  p = p,
  r = r,
  norm_rows = 1:r
)

# Print the differences between estimated and true parameters
print_fit_vs_truth(fit, truth, digits = 4)
