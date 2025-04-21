library(igraph)

# Define the data for the matrix as a vector (row-wise)
matrix_data <- c(0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 1, 1, 0, 1, 1,
                 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)

# Specify the number of rows and columns
num_rows <- 10
num_cols <- 10

# Create the matrix
A <- matrix(matrix_data, nrow = num_rows, ncol = num_cols, byrow = TRUE)

colnames(A) <- rownames(A) <- stringr::str_c("X",c(1:10))

# Print the matrix
print(A)


# Create graph
G <- graph_from_adjacency_matrix(A, mode = "undirected")

  # Eigen centrality
sort(eigen_centrality(G)$vector, decreasing = T)
  # Closeness centrality
sort(closeness(G), decreasing = T)
  # Betweeness centrality
sort(betweenness(G), decreasing = T)
