library(shiny)
library(leaflet)
library(tidyverse)
source('drawmap.R')
source('dataplotter.R')
source('tablemaker.R')

server <- function(input, output) {
  output$JPmap <- renderLeaflet({
    render.map( state = input$prefectureSelector)
  })


  output$stateplot <- renderPlot({
   state.plot(state = input$prefectureSelector)
  })

  output$table <- DT::renderDataTable({
    tablemaker(state = input$prefectureSelector,step = as.integer(input$lag))
  })
}

