library(reticulate)
np <- import('numpy')

# Load the .npy file
rmse_mat <- np$load('./data/RMSEmat.npy')

# Convert to R array
rmse_mat_r <- as.array(rmse_mat)

# Save the R array
save(rmse_mat_r, file = './data/RMSEmat.RData')

mae_mat <- np$load('./data/MAEmat.npy')
mae_mat_r <- as.array(mae_mat)
save(mae_mat_r, file = './data/MAEmat.RData')