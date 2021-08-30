args = commandArgs(trailingOnly=TRUE)
if (length(args)!=3 & length(args)!=7) {
  stop("Usage: Rscript --vanilla Rprogram Project_dir machine dev_id [Byte_low Byte_upper log_classes samples]", call.=FALSE)
}
project_dir = args[1]
machine = args[2]
dev_id = as.integer(args[3])

if (length(args)==7){
	Byte_low = as.integer(args[4])
	Byte_upper = as.integer(args[5])
	log_classes = as.integer(args[6])
	samples = as.integer(args[7]) 
} else{
	Byte_low = 10240
	Byte_upper = 104857600
	log_classes = 10
	samples = 100
}

infile_h2d = sprintf("%s/BenchOutputs/%s/cublasSet_Get/EMLR_to-0_from--1_MinB-%d_UpperStartB-%d_classes-%d_samples-%d.log", project_dir, machine, Byte_low, Byte_upper, log_classes, samples)
infile_d2h = sprintf("%s/BenchOutputs/%s/cublasSet_Get/EMLR_to--1_from-0_MinB-%d_UpperStartB-%d_classes-%d_samples-%d.log", project_dir, machine, Byte_low, Byte_upper, log_classes, samples)

h2d_dataset <- read.table(infile_h2d,  header = FALSE, sep = ",")
d2h_dataset <- read.table(infile_d2h,  header = FALSE, sep = ",")
#summary(h2d_dataset)
#summary(d2h_dataset)

# Remove intercept from model fitting 
inter_h2d <- subset(h2d_dataset$V4, h2d_dataset$V1 == 1)
simple_h2d_train_set <- subset(h2d_dataset, select=c("V4", "V1"))
simple_h2d_train_set$G_t <- c(I(simple_h2d_train_set$V4-inter_h2d))
#Calculate G with linear regression
G_h2d <- lm(G_t ~ -1 + V1 , data = subset(simple_h2d_train_set, simple_h2d_train_set$V1 != 1))
#Calculate slowdown 
G_h2d_bid <- lm(I(V3-inter_h2d) ~ -1 + V1 , data = subset(h2d_dataset, h2d_dataset$V1 != 1))
slowdown_h2d = G_h2d_bid$coefficients[1]/ G_h2d$coefficients[1]

inter_d2h <- (subset(d2h_dataset$V4, d2h_dataset$V1 == 1))
simple_d2h_train_set <- subset(d2h_dataset, select=c("V4", "V1"))
simple_d2h_train_set$G_t <- c(I(simple_d2h_train_set$V4-inter_d2h))
G_d2h <- lm(G_t ~ -1 + V1 , data = subset(simple_d2h_train_set, simple_d2h_train_set$V1 != 1))
G_d2h_bid <- lm(I(V3-inter_d2h) ~ -1 + V1 , data = subset(d2h_dataset, d2h_dataset$V1 != 1))
slowdown_d2h = G_d2h_bid$coefficients[1]/ G_d2h$coefficients[1]

cat("LogP H2D model:","\n")
cat("Intercept = ", inter_h2d,"\n")
cat("Coefficient =", G_h2d$coefficients[1],"\n")
cat("Slowdown =", slowdown_h2d,"\n")

cat("\nLogP D2H model:","\n")
cat("Intercept = ", inter_d2h,"\n")
cat("Coefficient =", G_d2h$coefficients[1],"\n")
cat("Slowdown =", slowdown_d2h,"\n")

outfile_h2d_name = sprintf("%s/BenchOutputs/%s/cublasSet_Get/model_%d_-1.out", project_dir, machine, dev_id)
outfile_d2h_name = sprintf("%s/BenchOutputs/%s/cublasSet_Get/model_-1_%d.out", project_dir, machine, dev_id)
outfile_h2d<-file(outfile_h2d_name)
outfile_d2h<-file(outfile_d2h_name)

writeLines(c(toString(inter_h2d),toString(G_h2d$coefficients[1]), toString(slowdown_h2d)), outfile_h2d)
close(outfile_h2d)

writeLines(c(toString(inter_d2h),toString(G_d2h$coefficients[1]), toString(slowdown_d2h)), outfile_d2h)
close(outfile_d2h)



