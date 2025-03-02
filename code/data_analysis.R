
# Look-up tables ----------------------------------------------------------


sheet1 <- readxl::read_xlsx("./data/Table 1 National Results MPI 2024.xlsx", skip = 7)
sheet1 <- sheet1[,2:7]
sheet1 <- na.omit(sheet1)
colnames(sheet1) <- c("iso","country","region","survey","year","mpi")
sheet1 <- sheet1[(tolower(sheet1$iso)) %in% substr(list.files("./results"),1,3),]

sheet_censored <- readxl::read_xlsx("./data/Table 1 National Results MPI 2024.xlsx", sheet = 2, skip = 7)
sheet_censored <- sheet_censored[,c(2:6,8:17)]
sheet_censored <- na.omit(sheet_censored)
colnames(sheet_censored) <- c("iso","country","region","survey","year",
                              'd_nutr', 'd_cm', 'd_educ', 'd_satt', 
                              'd_ckfl', 'd_sani','d_wtr','d_elct', 'd_hsg', 'd_asst')

sheet_raw <- readxl::read_xlsx("./data/Table 1 National Results MPI 2024.xlsx", sheet = 6, skip = 7)
sheet_raw <- sheet_raw[,c(2:6,8:17)]
sheet_raw <- na.omit(sheet_censored)
colnames(sheet_censored) <- c("iso","country","region","survey","year",
                              'd_nutr', 'd_cm', 'd_educ', 'd_satt', 
                              'd_ckfl', 'd_sani','d_wtr','d_elct', 'd_hsg', 'd_asst')

    ## Remember to change indicators order!

# Read and save the adjacency matrix for each country ---------------------
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

names(country_list) <- substr(list.files("./results"),1,3)


# Get the desired adjacency matrices
filter_graphs <- function(data, region = NULL,
                            c = 0,
                            censored = TRUE,
                            conservative = TRUE){
  
  if(!is.null(region)){
    if(region %in% sheet1$region){
      filtered.countries <- tolower(sheet1$iso[sheet1$region == region])
      data <- data[filtered.countries]
    }
  }
  
  
    
    adjacency_matrices <- sapply(data, function(x){
                              condition <- bquote(attr(y,'c') == .(c) &
                                                  attr(y,'censored') == .(censored) &
                                                  attr(y,"conservative") == .(conservative))
                              idx <- which(sapply(x, function(y) if(eval(condition)){TRUE}else{FALSE}) == TRUE)
                              return(x[idx])
    })
    
    return(adjacency_matrices)
}



X <- filter_graphs(country_list, conservative = FALSE)


indicators <- c('d_cm', 'd_nutr', 
                'd_satt', 'd_educ', 
                'd_elct', 'd_wtr', 'd_sani','d_hsg', 'd_ckfl', 'd_asst')

y <- as.matrix(Reduce("+", X)/length(X))

library(ggplot2)
melted_data <- reshape::melt(y)
melted_data
ggplot(melted_data, aes(x = X1, y = X2, fill = value)) +
  geom_tile(color = "black") +
  geom_text(aes(label = round(value,2)), color = "white", size = 4) +
  coord_fixed()




library(igraph)
degrees <- sapply(X, function(x){
  g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
  degree(g)
}
)
  
ggplot(reshape2::melt(degrees), aes(x = Var2, y = value, fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = scales::percent)

apply(degrees, MARGIN = 1, mean)







# Cliques -----------------------------------------------------------------

## These can be interpreted as "conditional" deprivation bundles

all_cliques_lists <- sapply(X, function(x){
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

ggplot(clique_counts_df[clique_counts_df$count>0.5,], aes(x = reorder(clique, count), y = count)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Keep the horizontal bars
  theme(axis.text.y = element_text(size = 6, angle = 0, hjust = 1)) + # Adjust size and angle
  labs(x = "Clique", y = "Count", title = "Clique Occurrences")


# Neighborhoods ----------------------------------------------------------

# Which indicators "isolates" an indicator
# The more important variables that explain the occurrence of a variable

neighborhood(g)


# Centrality --------------------------------------------------------------

ls <- lapply(X, function(x){
  g <- graph_from_adjacency_matrix(as.matrix(x), mode = "undirected")
  scores <- reshape2::melt(eigen_centrality(g)$vector)
  scores <- reshape2::melt(betweenness(g, directed = FALSE))
  scores$variable <- rownames(scores)
  headcounts <- sheet_censored[tolower(sheet_censored$iso) == attr(x,"country"),6:15]/100
  headcounts <- suppressMessages(reshape2::melt(headcounts[,indicators]))
  
  merged_df <- merge(scores, headcounts, by = "variable")
  colnames(merged_df) <- c("indicator","score","headcount")
  merged_df$country <- attr(x,"country")
  return(merged_df)
})

ggplot(do.call("rbind",ls)) + 
  geom_point(aes(x = score,y = headcount, color = indicator)) + 
  theme(legend.position = "none")


plot(eigen_centrality(g)$vector,headcounts[,indicators])
