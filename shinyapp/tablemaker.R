rmsemat <- readRDS('./data/RMSEmat.rds')
maemat  <- readRDS('./data/MAEmat.rds')
japan.names.vec <- sapply(readRDS("names_prefectures.rds"), 
                      function(x) x)

tablemaker <- function(state,step) {
  namesvec <- gsub("\\t", "", japan.names.vec)
  id <- which(namesvec == state)
  rmse <- apply(rmsemat[id,step,,],2,mean)
  mae <- apply(maemat[id,step,,], 2, mean)
  
  rmse.df <- data.frame(rbind(t(rmse),t(mae)))
  colnames(rmse.df) <- c('FBM','NOP','UFTS','MFTS','MFLTS')
  rownames(rmse.df) <- c('RMSE','MAE')
  rmse.df <- rmse.df %>% mutate_all(round,5)
  rmse.df
}