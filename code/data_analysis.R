library(igraph)

# Look-up tables ----------------------------------------------------------

## Read data from OPHI global MPI. Then, combine data with the estimated graphs

## global objects
## Countries in results
countries <- substr(list.files("./results"),1,3)
## Indicators in proper order
indicators <- c('d_nutr', 'd_cm', 
                'd_educ', 'd_satt',
                'd_ckfl', 'd_sani','d_wtr','d_elct', 'd_hsg', 'd_asst')

sheet1 <- readxl::read_xlsx("./data/Table 1 National Results MPI 2024.xlsx", skip = 7)
sheet1 <- sheet1[,2:7]
sheet1 <- na.omit(sheet1)
colnames(sheet1) <- c("iso","country","region","survey","year","mpi")
sheet1 <- sheet1[(tolower(sheet1$iso)) %in% countries,]
sheet1$iso <- tolower(sheet1$iso)

sheet_censored <- readxl::read_xlsx("./data/Table 1 National Results MPI 2024.xlsx", sheet = 2, skip = 7)
sheet_censored <- sheet_censored[,c(2:6,8:17)]
sheet_censored <- na.omit(sheet_censored)
colnames(sheet_censored) <- c("iso","country","region","survey","year",
                              indicators)
sheet_censored <- sheet_censored[(tolower(sheet_censored$iso)) %in% countries,]
sheet_censored$iso <- tolower(sheet_censored$iso)
### Reorder columns
sheet_censored <- sheet_censored[,c("iso","country","region","survey","year",
                                    indicators)]

sheet_raw <- readxl::read_xlsx("./data/Table 1 National Results MPI 2024.xlsx", sheet = 6, skip = 7)
sheet_raw <- sheet_raw[,c(2:6,8:17)]
sheet_raw <- na.omit(sheet_censored)
colnames(sheet_censored) <- c("iso","country","region","survey","year",
                              indicators)
sheet_raw <- sheet_raw[(tolower(sheet_raw$iso)) %in% countries,]
sheet_raw$iso <- tolower(sheet_raw$iso)
### Reorder columns
sheet_raw <- sheet_raw[,c("iso","country","region","survey","year",
                          indicators)]

# Read the adjacency matrix for each country ------------------------------
country_list <- lapply(list.files("./results"),
       function(x){
         country_iso <- substr(x, 1,3)
         
         file_names <- list.files(file.path("./results",x))
         
         adj_matrices <- lapply(file_names,
                                function(y){
                                  X <- read.table(file.path("./results",x,y))
                                  indicators <- c('d_cm', 'd_nutr', 
                                                  'd_satt', 'd_educ', 
                                                  'd_elct', 'd_wtr', 'd_sani','d_hsg', 'd_ckfl', 'd_asst') 
                                  rownames(X) <- colnames(X) <- indicators
                                  
                                  attr(X,"country") <- country_iso
                                  attr(X,"file name") <- y
                                  attr(X, "censored") <- if(grepl("_mpi_poor_", y)){TRUE}else{FALSE}
                                  attr(X, "conservative") <- if(grepl("_nconservative_", y)){FALSE}else{TRUE}
                                  attr(X,"c") <- as.numeric(stringr::str_extract(y, "(?<=e_c)(.*?)(?=\\.txt)"))
                                  attr(X, "data") <- "DMN" # Discrete Markov Network graph
                                  return(X)
                                })
         attr(adj_matrices,"country") <- country_iso
         attr(adj_matrices,"survey") <- gsub("[0-9]+","",sub(".*_", "", x))
         attr(adj_matrices,"year") <- gsub("[a-z]+","",sub(".*_", "", x))
         
         return(adj_matrices)
       })

names(country_list) <- countries


# Indicators global objects ------------------------------------------------

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


# Combine data ------------------------------------------------------------

for (country in names(country_list)){
  # check if survey and years match
  ## years
  year <- as.character(sheet1[sheet1$iso == country,"year"])
  if (grepl("-", year)) {
    parts <- strsplit(year, "-")[[1]]
    year <- paste0(substring(parts[1], 3), "-", substring(parts[2], 3))
  } else {
    year <- substring(year, 3)
  }
  stopifnot(year == attr(country_list[[country]],"year"))
  ## surveys
  survey <- tolower(as.character(sheet1[sheet1$iso == country,"survey"]))
  stopifnot(grepl(survey,attr(country_list[[country]],"survey")))
  
  # save world region
  attr(country_list[[country]],"region") <- as.character(sheet1[sheet1$iso == country,"region"])
  # save country full name
  attr(country_list[[country]],"country") <- as.character(sheet1[sheet1$iso == country,"country"])
  # save MPI results
  attr(country_list[[country]],"mpi") <- as.numeric(sheet1[sheet1$iso == country,"mpi"])
  # save censored indicators
  censored_indicators <- as.numeric(sheet_censored[sheet_censored$iso == country,indicators])
  names(censored_indicators) <- indicators
  attr(country_list[[country]],"censored") <- censored_indicators
  # save uncensored (raw) indicators
  raw_indicators <- as.numeric(sheet_raw[sheet_raw$iso == country,indicators])
  names(raw_indicators) <- indicators
  attr(country_list[[country]],"raw") <- raw_indicators
}


# Save data ---------------------------------------------------------------

saveRDS(country_list,"./app/data/data.rds")


# Get the desired adjacency matrices
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



X <- filter.DMN(country_list, region = "None",c= 5,conservative = FALSE, censored = FALSE)

y <- as.matrix(Reduce("+", X)/length(X))
y <- y[indicators, indicators]

library(ggplot2)
melted_data <- reshape::melt(y)
melted_data$X1 <- factor(stringr::str_replace_all(melted_data$X1,replacements),
                         levels = stringr::str_replace_all(indicators,replacements))
melted_data$X2 <- factor(stringr::str_replace_all(melted_data$X2,replacements),
                         levels = stringr::str_replace_all(indicators,replacements))
melted_data$X2 <- factor(melted_data$X2, levels = rev(levels(factor(melted_data$X2))))

ggplot(melted_data, aes(x = X1, y = X2, fill = value)) +
  geom_tile(color = "black") +
  geom_text(aes(label = round(value,2)), color = "white", size = 4) +
  scale_fill_gradient(limits = c(0, 1)) +
  labs(x = "", y = "", fill = "Proportion") +
  coord_fixed() 





degrees <- sapply(X, function(x){
  g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
  degree(g)
}
)
degrees <- degrees[indicators,]
rownames(degrees) <- stringr::str_replace_all(rownames(degrees), replacements)

zero_degree <- which(apply(degrees,MARGIN = 2, sum) == 0)

degrees <- degrees[,-zero_degree]

#### INDICATORS DISTRIBUTION
ggplot(reshape2::melt(degrees), aes(x = Var2, y = value, fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = scales::percent) + 
  scale_fill_manual(values = indicators_pallete) + 
  labs(x = "Countries", y = "Degree Distribution", fill = "Indicator") + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))



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

# Cliques -----------------------------------------------------------------

## These can be interpreted as "conditional" deprivation bundles

all_cliques_lists <- sapply(X, function(x){
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
                               count = clique_counts_table$Freq / length(X))
clique_counts_df$order <- sapply(strsplit(as.character(clique_counts_df[,1]), ","), length)

# Sort by count in descending order
clique_counts_df <- clique_counts_df[order(clique_counts_df$count, decreasing = TRUE), ]

# Clique order
clique_order <- 3

cliques_to_plot <- clique_counts_df[(clique_counts_df$order == clique_order),]
if (length(cliques_to_plot$count) > 15){
  cliques_to_plot <- cliques_to_plot[c(1:15),]
}

ggplot(cliques_to_plot, aes(x = reorder(clique, count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  coord_flip() +  # Keep the horizontal bars
  theme(axis.text.y = element_text(size = 6, angle = 0, hjust = 1)) + # Adjust size and angle
  labs(x = "Clique", y = "Occurences", title = "Clique Occurrences (in proportions)")


# Neighborhoods ----------------------------------------------------------

# Which indicators "isolates" an indicator
# The more important variables that explain the occurrence of a variable

x <- X[[6]]
x <- x[indicators,indicators]
g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
NB <- neighborhood(g)
names(NB) <- indicators
NB

# Centrality --------------------------------------------------------------

# Degree Centrality:
#   Definition: Counts the number of connections (edges) a node has. 
# Use Case: Simple and intuitive; good for identifying nodes with many direct connections. 
# Small Graph considerations: Very useful in small graphs. Easy to calculate, and gives a basic idea of node importance.
# Betweenness Centrality:
#   Definition: Measures how often a node lies on the shortest paths between other nodes.  
# Use Case: Identifies nodes that act as bridges or bottlenecks in the network. 
# Small Graph considerations: Very useful to find those nodes that connect different portions of a small graph. If the small graph is very connected, betweeness centrality can be less useful, as most nodes will be on many short paths.
# Closeness Centrality:
#   Definition: Measures the average shortest path distance from a node to all other nodes.  
# Use Case: Identifies nodes that are close to all other nodes in the network.
# Small Graph considerations: Good for finding nodes that can quickly reach other nodes. In small, tightly connected graphs, closeness centrality may not vary much between nodes. If your small graph is not well connected, it will show you the nodes that are the most efficient at reaching the other nodes.  
# Eigenvector Centrality:
#   Definition: Measures a node's influence based on the influence of its neighbors. Nodes with connections to highly influential nodes have higher eigenvector centrality. 
# Use Case: Identifies nodes that are connected to important nodes. 
# Small Graph considerations: Useful for understanding influence within the network. In small graphs, the influence of a few high-degree nodes can quickly propagate.

# Main Differences and Considerations for Small Graphs:
# 
# Focus:
# Degree centrality focuses on direct connections. 
# Betweenness centrality focuses on bridging roles. 
# Closeness centrality focuses on proximity. 
# Eigenvector focus on influence.


ls <- lapply(X, function(x){
  g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
  #scores <- eigen_centrality(g)$vector
  scores <- betweenness(g, directed = FALSE)
  names(scores) <- indicators
  return(scores)
})

plot.data <- apply(do.call("rbind",ls), MARGIN = 2, mean)

plot.data <- reshape2::melt(plot.data,value.name = "measure")
plot.data$indicator <- rownames(plot.data)

ggplot(plot.data, aes(x = reorder(indicator,measure), y = measure)) +
  geom_bar(stat = "identity", fill = "#a68580") +
  geom_text(aes(label = round(measure,2), hjust = 0)) +
  labs(title = "Centrality Measure",
       x = "Indicators",
       y = "Centrality level") + 
  coord_flip()







Y <- filter.DMN(country_list,country ="Namibia")
ctry <- names(Y)
if(attr(Y[[ctry]],"censored") == T){
  indicators_values <- attributes(country_list[[ctry]])$censored
  }else{
  indicators_values <- attributes(country_list[[ctry]])$raw
}
Y <- do.call("rbind", Y)
rownames(Y) <- colnames(Y)
Y <- Y[indicators,indicators]
graph <- graph_from_adjacency_matrix(as.matrix(Y), mode = "undirected")

V(graph)$color <- indicators_pallete
V(graph)$name <- stringr::str_replace_all(V(graph)$name, replacements)

plot(graph, 
     vertex.size = indicators_values,  # Set node sizes
     vertex.label = V(graph)$name, # Add node labels (optional)
     edge.width = 2,
     vertex.color = V(graph)$color,
     vertex.label.color = "black",
     main = "Conditional Dependencies \nbetween Poverty indicators",
     layout = layout_in_circle(graph))

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

centrality.measures <- c("Degree","Betweenness", "Closeness","Eigenvector")
cen <- "Betweeness"
if(cen == "Degree"){
  centrality <- centr_degree(graph)$res
  names(centrality) <- names(replacements)
  names(centrality) <- str_replace_all(names(centrality), replacements)
  } else if (cen == "Betweenness"){
    centrality <- betweenness(graph)
  } else if (cen == "Closeness"){
    centrality <- closeness(graph)
    centrality[is.na(centrality)] = 0
  } else {
    centrality <- eigen_centrality(graph)$vector
  }
centrality_df <- data.frame(variable = factor(names(centrality),
                                              levels = replacements),
                            value = as.numeric(centrality))
ggplot(centrality_df, aes(x = reorder(variable,value), y = value)) +
  geom_bar(stat = "identity", fill = "#a68580") +
  geom_text(aes(label = round(value,2), hjust = 0)) +
  labs(title = "Centrality Measure",
       x = "Indicators",
       y = "Centrality level") + 
  coord_flip()

