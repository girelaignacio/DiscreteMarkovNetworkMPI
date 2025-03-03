# Load libraries ----------------------------------------------------------
library(shiny)
library(igraph)
library(ggplot2)
library(reshape)

# Utils functions ---------------------------------------------------------

filter.DMN <- function(data, 
                       country = NULL, region = "None",
                       c = 0, censored = TRUE, conservative = TRUE){
  # Filter by country
  if(!is.null(country)){
    data <- data[sapply(data, function(x) attr(x,"country") == country)]
  }
  
  # Filter by region
  if(region != "None"){
    data <- data[sapply(data, function(x) attr(x,"region") == region)]
  }
  
  # Filter graphs adjacency matrices by penalization value (c), covariate 
  # (raw or censored), and criterion (conservative or non-conservative)
  adjacency_matrices <- sapply(data, function(x){
    condition <- bquote(attr(y,'c') == .(c) &
                          attr(y,'censored') == .(censored) &
                          attr(y,"conservative") == .(conservative))
    idx <- which(sapply(x, function(y) if(eval(condition)){TRUE}else{FALSE}) == TRUE)
    return(x[idx])
  }
  )
  return(adjacency_matrices)
}


# Global input objects ----------------------------------------------------

# global objects for filtering
countries.names <- sapply(data, function(x) attr(x,"country"))
covariate <- c("Uncensored", "Censored")
summary.measures <- c("degree","cliques","neighbourhoods")
# world region
world.region.names <- c("None","Latin America and the Caribbean",
                        "Sub-Saharan Africa",
                        "South Asia","Arab States","East Asia and the Pacific",
                        "Europe and Central Asia" )


# Define UI for application  ----------------------------------------------
ui <- fluidPage(
  # App title
  titlePanel("Markov Network Findings on Global MPI Patterns"),
  tabsetPanel(
    tabPanel("Aggregated Results",
             selectInput("summary", "Analysis and Measures", summary.measures),
             selectInput("region", "World Region", world.region.names),
             sliderInput("c", "Penalization parameter", value = 0, min = 0, max = 10),
             radioButtons("covariate", "Indicators", covariate),
             plotOutput("heatmap")
    ),
    tabPanel("Graph Plots",
             selectInput("country", "Country", countries.names),
             sliderInput("c", "Penalization parameter", value = 0, min = 0, max = 10),
             radioButtons("covariate", "Indicators", covariate)
    )
  )
)



# server ------------------------------------------------------------------
server <- function(input, output, session) {
  # COMPLETE
  # Load data
  data <- reactive({
    readRDS(normalizePath("./app/data.rds"))
  })
  # Input conditions that filters data
  covariate <- reactive({
    if (input$covariate == "Censored"){
      TRUE
      } else {
      FALSE
      }
    })
  
  # Filtered data
  X <- reactive({
    filter.DMN(data(), country = NULL, region = input$country, c = input$c,
               censored = covariate, conservative = FALSE)
  })
  
  ## Heatmap
  output$heatmap <- renderPlot({
    temp <- Reduce("+", X())/length(X())
    melted_data <- melt(temp)
    ggplot(melted_data, aes(x = X1, y = X2, fill = value)) +
      geom_tile(color = "black") +
      geom_text(aes(label = round(value,2)), color = "white", size = 4) +
      coord_fixed()
  })
}


# Run the application  ----------------------------------------------------
shinyApp(ui = ui, server = server)

