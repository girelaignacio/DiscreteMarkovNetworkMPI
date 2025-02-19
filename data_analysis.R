

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
                                  
                                  attr(X,"file name") <- y
                                  attr(X, "censored") <- if(grepl("_mpi_poor_", y)){TRUE}else{FALSE}
                                  attr(X, "conservative") <- if(grepl("_nconservative_", y)){FALSE}else{TRUE}
                                  attr(X,"c") <- as.numeric(stringr::str_extract(y, "(?<=e_c)(.*?)(?=\\.txt)"))
                                  attr(X, "data") <- "DMN" # Discrete Markov Network graph
                                  return(X)
                                })
         attr(adj_matrices,"country") <- country_iso
         return(adj_matrices)
       })


# Get the desired adjacency matrices
filter_matrices <- function(c = 0,
                            censored = TRUE,
                            conservative = TRUE){
    adjacency_matrices <- sapply(country_list, function(x){
                              condition <- bquote(attr(y,'c') == .(c) &
                                                  attr(y,'censored') == .(censored) &
                                                  attr(y,"conservative") == .(conservative))
                              idx <- which(sapply(x, function(y) if(eval(condition)){TRUE}else{FALSE}) == TRUE)
                              return(x[idx])
    })
    
    return(adjacency_matrices)
}

X <- filter_matrices()

indicators <- c('d_cm', 'd_nutr', 
                'd_satt', 'd_educ', 
                'd_elct', 'd_wtr', 'd_sani','d_hsg', 'd_ckfl', 'd_asst')

y <- as.matrix(Reduce("+", X)/46)

library(ggplot2)
melted_data <- reshape::melt(y)
melted_data
ggplot(melted_data, aes(x = X1, y = X2, fill = value)) +
  geom_tile(color = "black") +
  geom_text(aes(label = round(value,2)), color = "white", size = 4) +
  coord_fixed()
