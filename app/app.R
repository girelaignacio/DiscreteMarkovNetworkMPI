# Load libraries ----------------------------------------------------------
library(shiny)
library(igraph)
library(ggplot2)
library(reshape2)
library(scales)

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

# Load data
data <- readRDS('data/data.rds')


# Global input objects ----------------------------------------------------

# global objects for filtering
countries.names <- sapply(data, function(x) attr(x,"country"))
covariate <- c("Uncensored", "Censored")
summary.measures <- c("Proportions","Degrees","Cliques","neighbourhoods")
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
             sliderInput("c", "Penalization parameter", value = 0, min = 0, max = 9),
             radioButtons("covariate", "Indicators", covariate),
             plotOutput("plots")
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
    filter.DMN(data, country = NULL, region = input$region, c = input$c,
               censored = covariate(), conservative = FALSE)
  })
  
  ## Heatmap
  output$plots <- renderPlot({
    if (input$summary == "Proportions") {
      # X <- filter.DMN(data, country = NULL, region = input$region, c = input$c,
      #          censored = TRUE, conservative = FALSE)
      temp <- as.matrix(Reduce("+", X())/length(X()))
      melted_data <- melt(temp)
      melted_data$Var1 <- factor(melted_data$Var1)
      melted_data$Var2 <- factor(melted_data$Var2, levels = rev(levels(factor(melted_data$Var2))))
      ggplot(melted_data, aes(x = Var1, y = Var2, fill = value)) +
        geom_tile(color = "black") +
        geom_text(aes(label = round(value,2)), color = "white", size = 4) +
        coord_fixed()
    }
    else if (input$summary == "Degrees") {
    degrees <- sapply(X(), function(x){
      g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
      degree(g)
      })
    ggplot(melt(degrees), aes(x = Var2, y = value, fill = Var1)) + 
      geom_bar(stat = "identity", position = "fill") +
      scale_y_continuous(labels = percent)
    }
    else if (input$summary == "Cliques"){
      all_cliques_lists <- sapply(X(), function(x){
        g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
        components(g, mode = "weak")
        clq <- cliques(g, min = 2)
        clq_lists <- lapply(clq, function(y) sort(V(g)$name[y]))
        clique <- sapply(clq_lists, paste, collapse = ",")
      })
      all_cliques <- unlist(all_cliques_lists)
      clique_counts_table <- as.data.frame(table(all_cliques))
      
      # Convert to a data frame for easier handling
      clique_counts_df <- data.frame(clique = clique_counts_table$all_cliques,
                                     count = clique_counts_table$Freq / length(X))
      
      # Sort by count in descending order
      clique_counts_df <- clique_counts_df[order(clique_counts_df$count, decreasing = TRUE), ]
      
      ggplot(clique_counts_df[1:20,], aes(x = reorder(clique, count), y = count)) +
        geom_bar(stat = "identity") +
        coord_flip() +  # Keep the horizontal bars
        theme(axis.text.y = element_text(size = 6, angle = 0, hjust = 1)) + # Adjust size and angle
        labs(x = "Clique", y = "Count", title = "Clique Occurrences")
    }
  })
}


# Run the application  ----------------------------------------------------
shinyApp(ui = ui, server = server)

