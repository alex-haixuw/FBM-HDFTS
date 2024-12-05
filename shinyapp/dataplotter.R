library(reticulate)
library(rainbow)

np <- import("numpy")

japan.femaledata <- np$load("./data/jp_female_input.npy")
japan.names.vec <- sapply(readRDS("names_prefectures.rds"), 
                      function(x) x)

state.plot <- function(state) {
  gender.limits <- 1:101
  forecast.limits <- 1:50
  namesvec <- gsub("\\t", "", japan.names.vec)
  id <- which(namesvec == state)
  country.data <- japan.femaledata
  state.data <- country.data[,,id]
  
  matplot(t(state.data),type='l',col='gray86', 
          xlab = 'Age', ylab='Log mortality rate',
          main =paste0('Forecasted log mortality rate for ', state,' from 2013 - 2022'))
  
  forecast.data <- t(state.data[41:50,])
  colnames(forecast.data)<-1:10
  Res_forcasted_curves<-rainbow::fts(x=0:100, y=forecast.data, 
                                          xname='Age',yname='')
  rainbow::lines.fds(Res_forcasted_curves)
  
}

