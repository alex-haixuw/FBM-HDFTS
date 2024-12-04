library(shiny)
library(leaflet)


# load Rdata from a github repo
MAEmat <- readRDS(url('https://github.com/alex-haixuw/FBM-HDFTS/blob/2386b430beaebb760c410be7fab63c0c0fc40f49/shinyapp/MAEmat.RData'))


# Assuming the data is stored in a variable named 'MAEmat'
# Replace 'MAEmat' with the actual variable name if different
data <- MAEmat

# Define UI
ui <- fluidPage(
  titlePanel("MAEmat Data on Japan Map"),
  sidebarLayout(
    sidebarPanel(
      selectInput("value", "Value to display", choices = names(data))
    ),
    mainPanel(
      leafletOutput("map")
    )
  )
)

# Define server logic
server <- function(input, output) {
  output$map <- renderLeaflet({
    leaflet(data) %>%
      addTiles() %>%
      setView(lng = 138.2529, lat = 36.2048, zoom = 5) %>% # Center on Japan
      addCircleMarkers(
        lng = ~lon, lat = ~lat,
        radius = 5,
        color = ~colorNumeric("viridis", data[[input$value]])(data[[input$value]]),
        popup = ~paste(input$value, ":", data[[input$value]])
      )
  })
}

# Run the application 
shinyApp(ui = ui, server = server)