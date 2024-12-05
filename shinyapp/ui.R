library(shiny)
library(leaflet)
library(tidyverse)


ui <- fluidPage(
fluidRow(
    column(6, 
             selectInput("prefectureSelector",
                           "Prefecture",
                           sapply(readRDS("names_prefectures.rds"), 
                                  function(x) x)
               ),
                leafletOutput("JPmap"),
                

        #map 

        

        #Author info
            div(
             style = "display: flex; justify-content: center; align-items: center; padding-top:5%",
             h4(
               "Authors",
               br(),
               br(),
               "Haixu Wang",
               br(),
               "Dept of Math and Stats, University of Calgary, Canada",
               br(),
               br(),
               "Tianyu Guan",
               br(),
               "Dept of Math and Stats, York University, Canada",
               br(),
               br(),
               "Han Lin Shang",
               br(),
               "Department of Actuarial Studies and Business Analytics",
               br(),
               "Macquarie University, Australia")
             )
           
    ),
    column(6, 
      # plotting of predictions and time-wise comparison of rmse and mae 
      plotOutput("stateplot"),
      selectInput("lag",
                  "lag",
                  choices = c(1,2,3,4,5,6,7,8,9,10)
      ),
      DT::dataTableOutput("table")
      #Table of results 
    )
  )
)