library(igraph)

# Look-up tables ----------------------------------------------------------

## Read data from OPHI global MPI. Then, combine data with the estimated graphs

## global objects
## Countries in results
countries <- unique(substr(list.files("./results_stable"),1,3))
## Indicators in proper order
indicators <- c('d_nutr', 'd_cm', 
                'd_educ', 'd_satt',
                'd_ckfl', 'd_sani','d_wtr','d_elct', 'd_hsg', 'd_asst')

sheet1 <- readxl::read_xlsx("./utils/Table 1 National Results MPI 2024.xlsx", skip = 7)
sheet1 <- sheet1[,2:9]
sheet1 <- na.omit(sheet1)
colnames(sheet1) <- c("iso","country","region","survey","year","mpi","h","a")
sheet1 <- sheet1[(tolower(sheet1$iso)) %in% countries,]
sheet1$iso <- tolower(sheet1$iso)

sheet_censored <- readxl::read_xlsx("./utils/Table 1 National Results MPI 2024.xlsx", sheet = 2, skip = 7)
sheet_censored <- sheet_censored[,c(2:6,8:17)]
sheet_censored <- na.omit(sheet_censored)
colnames(sheet_censored) <- c("iso","country","region","survey","year",
                              indicators)
sheet_censored <- sheet_censored[(tolower(sheet_censored$iso)) %in% countries,]
sheet_censored$iso <- tolower(sheet_censored$iso)
### Reorder columns
sheet_censored <- sheet_censored[,c("iso","country","region","survey","year",
                                    indicators)]

sheet_raw <- readxl::read_xlsx("./utils/Table 1 National Results MPI 2024.xlsx", sheet = 6, skip = 7)
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
results <- sapply(list.files("./results_stable"),
       function(x){
         country_iso <- substr(x, 1,3)
         X <- read.table(file.path("./results_stable",x))
         print(X)
         rownames(X) <- colnames(X) <- indicators
         return(X)
         }, simplify = F, USE.NAMES = TRUE)

lookuptable <- sheet1
lookuptable <- merge(lookuptable, sheet_censored, by = c("iso","country","region","survey","year"))
lookuptable <- merge(lookuptable, sheet_raw, by = c("iso","country","region","survey","year"), suffixes = c("_censored","_uncensored"))

names(results) <- gsub(".txt","",names(results))


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

# Save data ---------------------------------------------------------------

saveRDS(results,"./app/data/data.rds")
saveRDS(lookuptable,"./app/data/lookup.rds")

data <- readRDS("./app/data/data.rds")
lookuptable <- readRDS("./app/data/lookup.rds")

# Get the desired adjacency matrices
filter_results <- function(X, 
                       country = NULL, region = "None",
                       censored = TRUE, conservative = TRUE){
  # Filter by country
  if(!is.null(country)){
    X <- X[grep(lookuptable$iso[which(lookuptable$country == country)], names(X)) ]
  }
  
  # Filter by region
  if(region != "None"){
    X <- X[substr(names(X),1,3) %in% lookuptable$iso[which(lookuptable$region == region)]]
  }
  
  # Filter by covariate (raw or mpi_poor)
  if (censored == TRUE){
    X <- X[grep("_mpi_poor_", names(X))]
  } else {
    X <- X[grep("_raw_", names(X))]
  }
  
  # Filter by criterion
  if (conservative == TRUE){
    X <- X[grep("_conserv", names(X))]
  } else {
    X <- X[grep("_nconser", names(X))]
  }
  return(X)
}



Y <- filter_results(results, country = "Argentina",conservative = FALSE, censored = TRUE)





























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

#degrees <- degrees[,-zero_degree]

#### INDICATORS DISTRIBUTION
ggplot(reshape2::melt(degrees), aes(x = Var2, y = value, fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = scales::percent) + 
  scale_fill_manual(values = indicators_pallete) + 
  labs(x = "Countries", y = "Degree Distribution", fill = "Indicator") + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))



#### AVERAGE DEGREE BY INDICATOR

degrees_df <- NULL
for (region in world.region.names){
  X <- filter_results(data,region = region, c= 0,conservative = TRUE, censored = TRUE)
  degrees <- sapply(X, function(x){
    g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
    degree(g)
  }
  )
  degrees <- degrees[indicators,]
  rownames(degrees) <- stringr::str_replace_all(rownames(degrees), replacements)

  prop_degrees <- apply(degrees, MARGIN = 1, mean)
  
  if(is.null(degrees_df)){
    degrees_df <- data.frame(variable = factor(names(prop_degrees),
                                               levels = replacements),
                             value = as.numeric(prop_degrees))
    colnames(degrees_df)[2] <- "World"
  } else {
    degrees_df <- merge(degrees_df, data.frame(variable = factor(names(prop_degrees),
                                                                 levels = replacements),
                                               value = as.numeric(prop_degrees)),
                        by = "variable")
    colnames(degrees_df)[which(colnames(degrees_df) == "value")] <- region
  }
}

degrees_df <- degrees_df[order(degrees_df$variable, replacements),]

# Create the bar plot
ggplot(degrees_df, aes(x = factor(variable), y = World)) +
  geom_bar(stat = "identity", fill = indicators_pallete) +
  #geom_text(aes(label = round(World,2),vjust = 2)) +
  labs(title = "Average Degree by Indicator",
       x = "Indicators",
       y = "Average degree") + 
  geom_point(aes(x = factor(variable), y = `Latin America and the Caribbean`), shape = 2, size = 4, stroke = 1) + 
  geom_point(aes(x = factor(variable), y = `Sub-Saharan Africa`), shape = 3, size = 4, stroke = 1) + 
  geom_point(aes(x = factor(variable), y = `Arab States`),shape = 4 , size = 4, stroke = 1) + 
  geom_point(aes(x = factor(variable), y = `East Asia and the Pacific`), shape = 5, size = 4, stroke = 1) + 
  geom_point(aes(x = factor(variable), y = `South Asia`), shape = 6, size = 4, stroke = 1) +
  geom_point(aes(x = factor(variable), y = `Europe and Central Asia`), shape = 1, size = 4, stroke = 1) +
  scale_shape_manual(name = "Region", # Set the title of the legend
                     values = c("Latin America and the Caribbean" = 2,
                                "Sub-Saharan Africa" = 3,
                                "Arab States" = 4,
                                "East Asia and the Pacific" = 5,
                                "South Asia" = 6,
                                "Europe and Central Asia" = 1))


plot.data <- reshape::melt(degrees_df)
colnames(plot.data) <- c("indicator","region","value")
ggplot(plot.data, aes(x = factor(indicator))) +
  geom_bar(data = subset(plot.data, region == "World"),
           aes(y = value),
           stat = "identity",
           fill = indicators_pallete) +
  geom_point(data = subset(plot.data, region != "World"),
             aes(y = value, shape = factor(region)),
             size = 4, stroke = 1) +
  theme(legend.position = "bottom")

  +
  labs(title = "Average Degree by Indicator",
       x = "Indicators",
       y = "Average degree",
       color = "Region",
       shape = "Region") +
  scale_shape_manual(name = "Region",
                     values = c("Latin America and the Caribbean" = 2,
                                "Sub-Saharan Africa" = 3,
                                "Arab States" = 4,
                                "East Asia and the Pacific" = 5,
                                "South Asia" = 6,
                                "Europe and Central Asia" = 1)) +
  scale_color_discrete(name = "Region")

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

