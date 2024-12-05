library(leaflet) 
library(sf)
library(dplyr)

japan.map <- readRDS("japanmap.Rds")

fill.color <- c("gray", "gold")

render.map <- function(state) {
  country = 'JAPAN'
  temp.map.df <- japan.map %>%
  mutate(
    NAME == if_else(rep(country == "JAPAN", nrow(.)), gsub("\\t", "", NAME), NAME),
    id = (NAME == state) + 1)

  fill <- fill.color[temp.map.df$id]
  selected.coords <- as.numeric(temp.map.df[temp.map.df$NAME == state, c("X", "Y")])[1:2]
  
  leaflet(temp.map.df) %>%
    addProviderTiles(providers$Esri.NatGeoWorldMap) %>%
    addPolygons(
      fillColor = fill, color = "black", fillOpacity = 0.5, weight = 1) %>% 
    addPopups(lng = selected.coords[1], lat = selected.coords[2], 
              popup = paste0("<b>", state, "</b>"))
}