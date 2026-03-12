n <- 1e8

t_total_0 <- proc.time()

t_sim_0 <- proc.time()
x <- rnorm(n)
t_sim <- proc.time() - t_sim_0

t_stats_0 <- proc.time()

x_mean <- mean(x)
x_sd <- sd(x)

z <- (x - x_mean) / x_sd
x_skew <- mean(z^3)
x_ex_kurt <- mean(z^4) - 3

t_stats <- proc.time() - t_stats_0
t_total <- proc.time() - t_total_0

cat("n               ", format(n, scientific = FALSE), "\n")
cat("mean            ", x_mean, "\n")
cat("sd              ", x_sd, "\n")
cat("skew            ", x_skew, "\n")
cat("excess kurtosis ", x_ex_kurt, "\n")
cat("\n")
cat("simulation time ", t_sim["elapsed"], "seconds\n")
cat("statistics time ", t_stats["elapsed"], "seconds\n")
cat("total time      ", t_total["elapsed"], "seconds\n")
