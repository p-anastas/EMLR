#Constants
dsize = 8

Gval_per_s <- function(value,time){
	return (value / (time * 1e9))
}

dgemm_flops <- function(M, N, K){
	return (M * K * (2 * N + 1))
}

dgemm_bytes <- function(M, N, K){
	return ((M * K + K * N + M * N * 2)*dsize)
}

dgemv_flops <- function(M, N){
	return (M * (2 * N + 1))
}

colMax <- function(data) sapply(data, max, na.rm = TRUE)
