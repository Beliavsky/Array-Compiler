n_sets <- 30L
n <- 1e7

t0_total <- proc.time()

means <- numeric(n_sets)

t0_loop <- proc.time()
for (i in seq_len(n_sets)) {
  x <- rnorm(n)
  means[i] <- mean(x)
}
t_loop <- proc.time() - t0_loop

t0_sd <- proc.time()
sd_means <- sd(means)
t_sd <- proc.time() - t0_sd

t_total <- proc.time() - t0_total

cat("number of sets           ", n_sets, "\n")
cat("sample size per set      ", format(n, scientific = FALSE), "\n")
cat("mean of means            ", mean(means), "\n")
cat("sd of means              ", sd_means, "\n")
cat("\n")
cat("generation/mean time     ", t_loop["elapsed"], "seconds\n")
cat("sd computation time      ", t_sd["elapsed"], "seconds\n")
cat("total elapsed time       ", t_total["elapsed"], "seconds\n")
