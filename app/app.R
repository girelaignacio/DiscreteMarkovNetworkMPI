# Load libraries ----------------------------------------------------------
library(shiny)
library(igraph)
library(ggplot2)
library(reshape2)
library(scales)
library(stringr)

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

# Get the desired adjacency matrices
filter_results <- function(X, 
                           country = NULL, region = "None",
                           c = 0, censored = TRUE, conservative = TRUE){
  # Filter by country
  if(!is.null(country)){
    X <- X[grep(lookuptable$iso[which(lookuptable$country == country)], names(X)) ]
    #data <- data[sapply(data, function(x) attr(x,"country") == country)]
  }
  
  # Filter by region
  if(region != "None"){
    X <- X[substr(names(X),1,3) %in% lookuptable$iso[which(lookuptable$region == region)]]
    #data <- data[sapply(data, function(x) attr(x,"region") == region)]
  }
  
  # Filter by penalization value c
  X <- X[grep(stringr::str_c("_c",c,".txt"), names(X))]
  
  # Filter by covariate (raw or mpi_poor)
  if (censored == TRUE){
    X <- X[grep("_mpi_poor_", names(X))]
  } else {
    X <- X[grep("_raw_", names(X))]
  }
  
  # Filter by criterion
  if (conservative == TRUE){
    X <- X[grep("_conservative_", names(X))]
  } else {
    X <- X[grep("_nconservative_", names(X))]
  }
  
  #names(X) <- NULL
  return(X)
}

# Load data ----------------------------------------------------------------

data <- readRDS('data/data.rds')

lookup <- readRDS('data/lookup.rds')


# Global input objects ----------------------------------------------------

# global objects for filtering
countries.names <- lookup$country
#names(countries.names) <- NULL
covariate <- c("Uncensored", "Censored")
summary.measures <- c("Edge occurrences","Degree Distribution","Average Degree","Cliques", "Centrality")
centrality.measures <- c("Degree","Betweenness", "Closeness","Eigenvector")
# world region
world.region.names <- c("None","Latin America and the Caribbean",
                        "Sub-Saharan Africa",
                        "South Asia","Arab States","East Asia and the Pacific",
                        "Europe and Central Asia" )

# Indicators order, replacements, and pallettes
indicators <- c('d_nutr', 'd_cm', 
                'd_educ', 'd_satt',
                'd_ckfl', 'd_sani','d_wtr','d_elct', 'd_hsg', 'd_asst')

replacements <- c(
  "d_nutr" = "NU",
  "d_cm" = "CM",
  "d_educ" = "YS",
  "d_satt" = "SA",
  "d_ckfl" = "CF",
  "d_sani" = "SN",
  "d_wtr" = "DW",
  "d_elct" = "EC",
  "d_hsg" = "HO",
  "d_asst" = "AS"
)

indicators_pallete <- c("#962b21", # d_nutr
                        "#652525", # d_cm
                        "#c6a9ab", # d_educ
                        "#a68580", # d_satt
                        "#afc4d1", # d_ckfl
                        "#7d9eb6", # d_sani
                        "#5e8199", # d_wtr
                        "#3f6781", # d_elec
                        "#174d68", # d_hsg
                        "#00384f") # d_asst


# Define UI for application  ----------------------------------------------
ui <- fluidPage(
  # App title
  titlePanel("Markov Network Findings on Global MPI Patterns"),
  tabsetPanel(
    tabPanel("Aggregated analysis",
             div(
               style = "display:flex; align-items:flex-start",
               wellPanel( #~~ Sidebar ~~#
                 style = "overflow-y: auto; position:fixed; width:300px; top:8 bottom:0",
             selectInput("summary", "Analysis and Measures", summary.measures),
             selectInput("region", "World Region", world.region.names),
             sliderInput("c", "Penalization parameter", value = 0, min = 0, max = 9),
             radioButtons("covariate", "Indicators", covariate, selected = "Uncensored"),
             conditionalPanel(
               condition = "input.summary == 'Cliques'",
               sliderInput("clique_order", "Clique Order:",
                           min = 2, max = 5, value = 2, step = 1)),
             conditionalPanel(
               condition = "input.summary == 'Centrality'",
               radioButtons("centrality_measure", "Measure", centrality.measures, selected = "Degree"))
             ),
             div( #~~ Main panel ~~#
               style = "flex-grow:1; resize:horizontal; overflow: hidden; position:relative; margin-left: 310px",
               plotOutput("plots")
             )
          )
    ),
    tabPanel("Analysis by country",
             div(
               style = "display:flex; align-items:flex-start",
               wellPanel( #~~ Sidebar ~~#
                 style = "overflow-y: auto; position:fixed; width:300px; top:8 bottom:0",
                 selectInput("country", "Country", countries.names),
                 sliderInput("c", "Penalization parameter", value = 0, min = 0, max = 9),
                 radioButtons("covariate", "Indicators", covariate),
                 selectInput("centrality", "Centrality Measure", centrality.measures)
               ),
               div( #~~ Main panel ~~#
                 style = "flex-grow:1; resize:horizontal; overflow: hidden; position:relative; margin-left: 310px",
                 fluidRow(
                   column(6, plotOutput("graph")),
                   column(6, plotOutput("headcounts"))
                 ),
                 fluidRow(
                   column(6, plotOutput("centrality")),
                   column(6, verbatimTextOutput("country_briefing"))
                 )
               ) # end of second div
             ) # end of first div
            ) # end of tabPanel
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
    filter_results(data, country = NULL, region = input$region, c = input$c,
               censored = covariate(), conservative = FALSE)
  })
  
  ## Heatmap
  output$plots <- renderPlot({
    if (input$summary == "Edge occurrences") {
      # X <- filter.DMN(data, country = NULL, region = input$region, c = input$c,
      #          censored = TRUE, conservative = FALSE)
      temp <- as.matrix(Reduce("+", X())/length(X()))
      temp <- temp[indicators, indicators]
      melted_data <- melt(temp)
      melted_data$Var1 <- factor(str_replace_all(melted_data$Var1,replacements),
                               levels = str_replace_all(indicators,replacements))
      melted_data$Var2 <- factor(str_replace_all(melted_data$Var2,replacements),
                               levels = str_replace_all(indicators,replacements))
      melted_data$Var2 <- factor(melted_data$Var2, levels = rev(levels(factor(melted_data$Var2))))
      #melted_data$Var1 <- factor(melted_data$Var1)
      #melted_data$Var2 <- factor(melted_data$Var2, levels = rev(levels(factor(melted_data$Var2))))
      ggplot(melted_data, aes(x = Var1, y = Var2, fill = value)) +
        geom_tile(color = "black") +
        geom_text(aes(label = round(value,2)), color = "white", size = 4) +
        scale_fill_gradient(limits = c(0, 1)) +
        labs(x = "", y = "", fill = "Proportion", title = "Edge occurrence (in proportions)") +
        coord_fixed() 
    }
    else if (input$summary == "Degree Distribution") {
    degrees <- sapply(X(), function(x){
      g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
      degree(g)
      })
    degrees <- degrees[indicators,]
    rownames(degrees) <- stringr::str_replace_all(rownames(degrees), replacements)
    zero_degree <- which(apply(unname(degrees),MARGIN = 2, sum) == 0)
    if (!(is.integer(zero_degree) && length(zero_degree) == 0)){
      degrees <- degrees[,-zero_degree]
      }
    
    ## Plot
    plt_data <- reshape2::melt(degrees)
    plt_data$Var2 <- substr(plt_data$Var2,1,3)
    ggplot(plt_data, aes(x = Var2, y = value, fill = Var1)) + 
      geom_bar(stat = "identity", position = "fill") +
      scale_y_continuous(labels = scales::percent) + 
      scale_fill_manual(values = indicators_pallete) + 
      labs(x = "Countries", y = "Degree Distribution", fill = "Indicator") + 
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
    } 
    else if (input$summary == "Average Degree"){
      degrees <- sapply(X(), function(x){
        g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
        degree(g)
      })
      degrees <- degrees[indicators,]
      rownames(degrees) <- stringr::str_replace_all(rownames(degrees), replacements)
      
      #### AVERAGE DEGREE BY INDICATOR
      prop_degrees <- apply(degrees, MARGIN = 1, mean)
      degrees_df <- data.frame(variable = factor(names(prop_degrees),
                                                 levels = replacements),
                               value = as.numeric(prop_degrees))
      
      # Create the bar plot
      ggplot(degrees_df, aes(x = factor(variable), y = value)) +
        geom_bar(stat = "identity", fill = indicators_pallete) +
        geom_text(aes(label = round(value,2),vjust = 2)) +
        labs(title = "Average Degree by Indicator",
             x = "Indicators",
             y = "Average degree")
    }
    else if (input$summary == "Cliques"){
      all_cliques_lists <- sapply(X(), function(x){
        x <- x[indicators, indicators]
        rownames(x) <- colnames(x) <- stringr::str_replace_all(rownames(x), replacements)
        g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
        clq <- cliques(g, min = 2)
        clq_lists <- lapply(clq, function(y) sort(V(g)$name[y]))
        clique <- sapply(clq_lists, paste, collapse = ",")
      })
      all_cliques <- unlist(all_cliques_lists)
      clique_counts_table <- as.data.frame(table(all_cliques))
      
      # Convert to a data frame for easier handling
      clique_counts_df <- data.frame(clique = clique_counts_table$all_cliques,
                                     count = clique_counts_table$Freq / length(X()))
      clique_counts_df$order <- sapply(strsplit(as.character(clique_counts_df[,1]), ","), length)
      
      # Sort by count in descending order
      clique_counts_df <- clique_counts_df[order(clique_counts_df$count, decreasing = TRUE), ]
      
      # Clique order
      clique_order <- input$clique_order
      
      cliques_to_plot <- clique_counts_df[(clique_counts_df$order == clique_order),]
      if (length(cliques_to_plot$count) > 15){
        cliques_to_plot <- cliques_to_plot[c(1:15),]
      }
      
      ggplot(cliques_to_plot, aes(x = reorder(clique, count), y = count)) +
        geom_bar(stat = "identity", fill = "#174d68", color = "black") +
        coord_flip() +  # Keep the horizontal bars
        theme(axis.text.y = element_text(size = 10, angle = 0, hjust = 1)) + # Adjust size and angle
        labs(x = "Clique", y = "Occurences", title = "Clique Occurrences (in proportions)")
    } 
    else if (input$summary == "Centrality"){
      # Centrality indicators (average)
      # All measure in list
      selected_measure <- input$centrality_measure
      measures_list <- lapply(X(), function(x){
        g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
        if (selected_measure == "Degree"){
          scores <- degree(g)
        } else if (selected_measure == "Betweenness") {
          scores <- betweenness(g, directed = FALSE)
        } else if (selected_measure == "Closeness"){
          scores <- closeness(g)
          scores[which(is.na(scores))] <- 0
        } else if (selected_measure == "Eigenvector"){
          scores <- eigen_centrality(g)$vector
        }
        names(scores) <- indicators
        return(scores)
      })
      plot.data <- apply(do.call("rbind",measures_list), MARGIN = 2, mean)
      plot.data <- melt(plot.data,value.name = "measure")
      
      rownames(plot.data) <- str_replace_all(rownames(plot.data), replacements)
      
      plot.data$indicator <- rownames(plot.data)
      
      
      ggplot(plot.data, aes(x = reorder(indicator,measure), y = measure)) +
        geom_bar(stat = "identity", fill = "#a68580") +
        geom_text(aes(label = round(measure,2), hjust = -0.5)) +
        labs(title = "Centrality Measure",
             x = "Indicators",
             y = "Centrality level") + 
        coord_flip()
    }
  })
  
  Y <- reactive({
    filter_results(data, country = input$country, c = input$c,
               censored = covariate(), conservative = FALSE)
  })
  
  output$graph <- renderPlot({
    ctry <- names(Y())
    # if(attr(Y()[[ctry]],"censored") == T){
    #   indicators_values <- attributes(data[[ctry]])$censored
    # }else{
    #   indicators_values <- attributes(data[[ctry]])$raw
    # }
    Y <- do.call("rbind", Y())
    rownames(Y) <- colnames(Y)
    Y <- Y[indicators,indicators]
    #graph <- graph_from_adjacency_matrix(as.matrix(Y), mode = "undirected")
    # plot(graph,
    #      vertex.size = indicators_values,  # Set node sizes
    #      vertex.label = stringr::str_replace_all(V(graph)$name, replacements), # Add node labels (optional)
    #      edge.width = 2,
    #      vertex.color = indicators_pallete,
    #      vertex.label.color = "black",
    #      main = "Conditional Dependencies \nbetween Poverty indicators",
    #      layout = layout_in_circle(graph))
    qgraph(input = as.matrix(Y), color = indicators_pallete, layout = "circle",
          labels = str_replace_all(colnames(Y), replacements))
  }) # end of output$graph
  
  output$headcounts <- renderPlot({
    ctry <- substr(names(Y()),1,3)
    # if(attr(Y()[[ctry]],"censored") == T){
    #   indicators_values <- attributes(data[[ctry]])$censored
    # }else{
    #   indicators_values <- attributes(data[[ctry]])$raw
    # }
    
    if (covariate() == TRUE){
      indicators_values <- lookup[lookup$iso == ctry, grep("_censored", colnames(lookup))]
    } else {
      indicators_values <- lookup[lookup$iso == ctry, grep("_uncensored", colnames(lookup))]
    }
    names(indicators_values) <- gsub(pattern = "_censored",
                                     replacement = "", 
                                     x = names(lookup[lookup$iso == ctry, grep("_censored", colnames(lookup))]))
    names(indicators_values) <- stringr::str_replace_all(names(indicators_values), replacements)
    
    headcounts_df <- data.frame(variable = factor(names(indicators_values),
                                                  levels = replacements),
                                value = as.numeric(indicators_values))
    
    # Create the bar plot
    ggplot(headcounts_df, aes(x = factor(variable), y = value)) +
      geom_bar(stat = "identity", fill = indicators_pallete) +
      geom_text(aes(label = round(value,2),vjust = -1.5)) +
      ylim(0,100) + 
      labs(title = "Headcount Ratios",
           x = "Indicators",
           y = "Headcount Ratio (%)")
  }) # end of output$headcounts
  
  output$centrality <- renderPlot({
    y <- do.call("rbind", Y())
    rownames(y) <- colnames(y)
    y <- y[indicators,indicators]
    graph <- graph_from_adjacency_matrix(as.matrix(y), mode = "undirected")
    if(input$centrality == "Degree"){
      centrality <- centr_degree(graph)$res
      names(centrality) <- names(replacements)
      names(centrality) <- str_replace_all(names(centrality), replacements)
    } else if (input$centrality == "Betweenness"){
      centrality <- betweenness(graph)
      names(centrality) <- str_replace_all(names(centrality), replacements)
    } else if (input$centrality == "Closeness"){
      centrality <- closeness(graph)
      centrality[is.na(centrality)] = 0
      names(centrality) <- str_replace_all(names(centrality), replacements)
    } else {
      centrality <- eigen_centrality(graph)$vector
      names(centrality) <- str_replace_all(names(centrality), replacements)
    }
    centrality_df <- data.frame(variable = factor(names(centrality),
                                                  levels = replacements),
                                value = as.numeric(centrality))
    ggplot(centrality_df, aes(x = reorder(variable,value), y = value)) +
      geom_bar(stat = "identity", fill = "#a68580") +
      geom_text(aes(label = round(value,2), hjust = - 0.3)) +
      labs(title = "Centrality Measure",
           x = "Indicators",
           y = "Centrality level") + 
      coord_flip()
  })
  
  output$country_briefing <- renderText({
    ctry <- substr(names(Y()),1,3)
    paste("Country Attributes: ",
          "\n",
          sprintf("Country: %s", lookup$country[lookup$iso == ctry]),
          "\n",
          sprintf("Region: %s", lookup$region[lookup$iso == ctry]),
          "\n",
          sprintf("MPI: %.3f", round(lookup$mpi[lookup$iso == ctry],3)),
          "\n",
          sprintf("H: %.3f%%", round(lookup$h[lookup$iso == ctry],3)),
          "\n",
          sprintf("A: %.3f%%", round(lookup$a[lookup$iso == ctry],3)),
          "\n",
          sprintf("Survey: %s", lookup$survey[lookup$iso == ctry]),
          "\n",
          sprintf("Year: %s",  lookup$year[lookup$iso == ctry])
          )
  })
}


# Run the application  ----------------------------------------------------
shinyApp(ui = ui, server = server)

