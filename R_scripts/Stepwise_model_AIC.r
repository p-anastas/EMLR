#install.packages(c("MASS"))
library(MASS)

args = commandArgs(trailingOnly=TRUE)
if (length(args)!=8 & length(args)!=10 & length(args)!=12) {
  stop("Usage: Rscript --vanilla Rprogram Project_dir system dev_id func Dmin [Mmax] Nmax [Kmax] [Mstep] Nstep [Kstep] bench_lim", call.=FALSE)
}
project_dir = args[1]
machine = args[2]
dev_id = as.integer(args[3])
func = args[4]
Dmin = as.integer(args[5])

if (length(args)==12){
	Mmax = as.integer(args[6])
	Nmax = as.integer(args[7])
	Kmax = as.integer(args[8])
	Mstep = as.integer(args[9])
	Nstep = as.integer(args[10])
	Kstep = as.integer(args[11])
	bench_lim = as.integer(args[12]) 

	infile = sprintf("%s/BenchOutputs/%s/cublas%s/EMLR_dev-%d_minDim-%d_max-%d-%d-%d_step-%d-%d-%d_lim-%d.log", project_dir, machine, func, dev_id, Dmin, Mmax, Nmax, Kmax, Mstep, Nstep, Kstep, bench_lim)
	gemm_gpu_df <- read.table(infile,  header = FALSE, sep = ",")
	#summary(gemm_gpu_df)

	# Remove very small values from model fitting 
	gemm_gpu_df_clean <- subset(gemm_gpu_df, gemm_gpu_df$V1 > 255 & gemm_gpu_df$V2 > 255 & gemm_gpu_df$V3 > 255)

	# V4 = t_Avg ITER, V5 = t_Min ITER, V6 = t_Max ITER
	gemm_gpu_train_set <- (subset(gemm_gpu_df_clean, select=c("V1", "V2", "V3", "V4", "V5", "V6")))
	gemm_gpu_train_set$SqT <- c(I(gemm_gpu_train_set$V1*gemm_gpu_train_set$V2))
	gemm_gpu_train_set$Exec_T <- c(I(gemm_gpu_train_set$V4))

	# The analytic representation model 
	analytic_model <-  lm(Exec_T ~SqT:V3 - 1, data = gemm_gpu_train_set)

	# Stepwise regression model. Use the AIC criterion with k = 10 ( normally is 2, but we want a larger 'punishment' for variable addition. 
	step_model <- stepAIC(analytic_model, scope = list(upper = ~SqT:V3 + SqT + V1:V3 + V1 + V3 - 1, lower = ~1), direction = "both", k = 10, trace = TRUE) # FALSE

	step_model$anova
	print(summary(step_model))
	pdf_stats = sprintf("%s/BenchOutputs/%s/cublas%s/EMLR_dev-%d_minDim-%d_max-%d-%d-%d_step-%d-%d-%d_lim-%d_aic.pdf", project_dir, machine, func, dev_id, Dmin, Mmax, Nmax, Kmax, Mstep, Nstep, Kstep, bench_lim)
	pdf(pdf_stats)
	par(mfrow = c(2, 2))
	plot(step_model)
	dev.off()
	#step_model$coefficients

	outlist <- c(0,0,0,0,0,0,0,0)

	if ('(Intercept)' %in% names(step_model$coefficients)) { if (step_model$coefficients['(Intercept)'] >=0) {outlist[1] = step_model$coefficients['(Intercept)'] }}

	if ('V1' %in% names(step_model$coefficients)) { if (step_model$coefficients['V1'] >=0) {outlist[2] = step_model$coefficients['V1'] }}
	if ('V3' %in% names(step_model$coefficients)) { if (step_model$coefficients['V3'] >=0) {outlist[4] = step_model$coefficients['V3'] }}

	if ('SqT' %in% names(step_model$coefficients)) { if (step_model$coefficients['SqT'] >=0) {outlist[5] = step_model$coefficients['SqT'] }}

	if ('V1:V3' %in% names(step_model$coefficients)) { if (step_model$coefficients['V1:V3'] >=0) {outlist[6] = step_model$coefficients['V1:V3'] }}
	if ('V3:V1' %in% names(step_model$coefficients)) { if (step_model$coefficients['V3:V1'] >=0) {outlist[6] = step_model$coefficients['V3:V1'] }}

	if ("SqT:V3" %in% names(step_model$coefficients)) { if (step_model$coefficients['SqT:V3'] >=0) {outlist[8] = step_model$coefficients["SqT:V3"] }}
	if ("V3:SqT" %in% names(step_model$coefficients)) { if (step_model$coefficients['V3:SqT'] >=0) {outlist[8] = step_model$coefficients["V3:SqT"] }}

} else if (length(args)==10) {
	Mmax = as.integer(args[6])
	Nmax = as.integer(args[7])
	Mstep = as.integer(args[8])
	Nstep = as.integer(args[9])
	bench_lim = as.integer(args[10]) 

	infile = sprintf("%s/BenchOutputs/%s/cublas%s/EMLR_dev-%d_minDim-%d_max-%d-%d_step-%d-%d_lim-%d.log", project_dir, machine, func, dev_id, Dmin, Mmax, Nmax, Mstep, Nstep, bench_lim)
	gemv_gpu_df <- read.table(infile,  header = FALSE, sep = ",")
	#summary(gemv_gpu_df)

	# Remove very small values from model fitting 
	gemv_gpu_df_clean <- subset(gemv_gpu_df, gemv_gpu_df$V1 > 255 & gemv_gpu_df$V2 > 255)	

	# V3 = t_Avg ITER, V4 = t_Min ITER, V5 = t_Max ITER
	gemv_gpu_train_set <- (subset(gemv_gpu_df_clean, select=c("V1", "V2", "V3", "V4", "V5")))
	gemv_gpu_train_set$Exec_T <- c(I(gemv_gpu_train_set$V3))

	# The analytic representation model 
	analytic_model <-  lm(Exec_T ~V1:V2 - 1, data = gemv_gpu_train_set)

	# Stepwise regression model. Use the AIC criterion with k = 10 ( normally is 2, but we want a larger 'punishment' for variable addition. 
	step_model <- stepAIC(analytic_model, scope = list(upper = ~V1:V2 + V1 + V2 - 1, lower = ~1), direction = "both", k = 10, trace = TRUE) # FALSE

	step_model$anova
	print(summary(step_model))
	pdf_stats = sprintf("%s/BenchOutputs/%s/cublas%s/EMLR_dev-%d_minDim-%d_max-%d-%d_step-%d-%d_lim-%d_aic.pdf", project_dir, machine, func, dev_id, Dmin, Mmax, Nmax, Mstep, Nstep, bench_lim)
	pdf(pdf_stats)
	par(mfrow = c(2, 2))
	plot(step_model)
	dev.off()
	#step_model$coefficients

	outlist <- c(0,0,0,0)

	if ('(Intercept)' %in% names(step_model$coefficients)) { if (step_model$coefficients['(Intercept)'] >=0) {outlist[1] = step_model$coefficients['(Intercept)'] }}

	if ('V1' %in% names(step_model$coefficients)) { if (step_model$coefficients['V1'] >=0) {outlist[2] = step_model$coefficients['V1'] }}
	if ('V2' %in% names(step_model$coefficients)) { if (step_model$coefficients['V2'] >=0) {outlist[3] = step_model$coefficients['V2'] }}

	if ('V1:V2' %in% names(step_model$coefficients)) { if (step_model$coefficients['V1:V2'] >=0) {outlist[4] = step_model$coefficients['V1:V2'] }}
	if ('V2:V1' %in% names(step_model$coefficients)) { if (step_model$coefficients['V2:V1'] >=0) {outlist[4] = step_model$coefficients['V2:V1'] }}

} else if (length(args)==8) {
	Nmax = as.integer(args[6])
	Nstep = as.integer(args[7])
	bench_lim = as.integer(args[8]) 

	infile = sprintf("%s/BenchOutputs/%s/cublas%s/EMLR_dev-%d_min-%d_max-%d_step-%d_lim-%d.log", project_dir, machine, func, dev_id, Dmin, Nmax, Nstep, bench_lim)
	axpy_gpu_df <- read.table(infile,  header = FALSE, sep = ",")
	#summary(gemv_gpu_df)

	# Remove very small values from model fitting 
	axpy_gpu_df_clean <- subset(axpy_gpu_df, axpy_gpu_df$V1 > 255 )	

	# V2 = t_Avg ITER, V3 = t_Min ITER, V4 = t_Max ITER
	axpy_gpu_train_set <- (subset(axpy_gpu_df_clean, select=c("V1", "V2", "V3", "V4")))
	axpy_gpu_train_set$Exec_T <- c(I(axpy_gpu_train_set$V3))

	# The analytic representation model 
	step_model <-  lm(Exec_T ~V1, data = axpy_gpu_train_set)

	step_model$anova
	print(summary(step_model))
	pdf_stats = sprintf("%s/BenchOutputs/%s/cublas%s/EMLR_dev-%d_min-%d_max-%d_step-%d_lim-%d_aic.pdf", project_dir, machine, func, dev_id, Dmin, Nmax, Nstep, bench_lim)
	pdf(pdf_stats)
	par(mfrow = c(2, 2))
	plot(step_model)
	dev.off()
	#step_model$coefficients

	outlist <- c(0,0)

	if ('(Intercept)' %in% names(step_model$coefficients)) { if (step_model$coefficients['(Intercept)'] >=0) {outlist[1] = step_model$coefficients['(Intercept)'] }}

	if ('V1' %in% names(step_model$coefficients)) { if (step_model$coefficients['V1'] >=0) {outlist[2] = step_model$coefficients['V1'] }}

} else {
  stop("Invalid combination of arguments and functions", call.=FALSE)
}

# Store for C use. 
outfile_name = sprintf("%s/BenchOutputs/%s/cublas%s/model_dev-%d.out", project_dir, machine, func, dev_id)
outfile<-file(outfile_name)
writeLines(sapply(outlist,toString), outfile)
close(outfile)


quit()


