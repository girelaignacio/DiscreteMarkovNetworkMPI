# Load libraries
library(ggplot2)

# Figure Y ----------------------------------------------------------------

globalMPI_table <- readxl::read_xlsx("./utils/Table 1 National Results MPI 2024.xlsx", skip = 7)
globalMPI_table <- globalMPI_table[,2:9]
globalMPI_table <- na.omit(globalMPI_table)
colnames(globalMPI_table) <- c("iso","country","region","survey","year","mpi","h","a")
countries_sample <- toupper(substr(list.files("./processed_data"),1,3))

in_sample <- ifelse(globalMPI_table$iso %in% countries_sample, 1, 0)

plot.data <- cbind(globalMPI_table, in_sample)

ggplot(plot.data, aes(y=mpi, x = region, fill = as.factor(in_sample))) + 
  geom_boxplot()

# Figure X ----------------------------------------------------------------

in_sample <- ifelse(globalMPI_table$iso %in% countries_sample, 1, 0)
in_sample <- cbind(globalMPI_table[2:3],in_sample) 
in_sample$count <- 1

plot.data <- merge(aggregate(in_sample ~ region, in_sample, sum),
                   aggregate(count ~ region, in_sample, sum), 
                   by = c("region"))
plot.data$region <- gsub("and","and\n",plot.data$region)
plot.data$region <- gsub("haran","haran\n",plot.data$region)

ggplot(plot.data) +
  geom_bar(stat = "identity",  aes(x = region, y = count, fill = "red"), width = 0.5) + # 'alpha' for some transparency
  geom_bar(stat = "identity", aes(x = region, y = in_sample, fill = "steelblue"), width = 0.5) +
  theme_minimal() + xlab("World Region") + ylab("Number of countries") +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 30, vjust = 0.89))


# Get all files
files <- list.files("./processed_data")

# Get all the possible combinations of deprivations 
# and calculate their empirical probability (see if positive or not)
frequencies <- lapply(files, FUN = function(file){
  print(file)
  # Read data
  data <- read.csv(stringr::str_c("./processed_data/",file))
  # Frequency tables
  frequencies <- reshape2::melt(prop.table(table(data[,-1])))
  frequencies$positive <- ifelse(frequencies$value > 0, 1, 0)
  frequencies$country <- factor(file)
  return(frequencies)}
)
# convert list into data frame
frequencies <- do.call("rbind", frequencies)

# Agregate frequencies by positive and non-positive
plt.data <- aggregate(value ~ positive + country, data = frequencies, 
                      FUN = function(x){length(x)})

plt.data$country <- substr(plt.data$country,1,3)
plt.data$positive <- ifelse(plt.data$positive == "0", "Zero","Positive")
# Plot!
ggplot(plt.data, aes(fill = factor(positive), y = value, x = factor(country))) +
  geom_bar(position = "stack", stat = "identity") +
  labs(
       x = "Countries",
       y = "Deprivation profiles",
       fill = "Probabilities") +
  #theme_bw() + # A clean theme
  scale_fill_manual(values = c("Zero" = "darkblue", "Positive" = "skyblue")) +
  theme(axis.text.y=element_blank(),
        legend.position="bottom",
        axis.text.x = element_text(angle = 45, hjust = 1,size = 13),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15),
        axis.title = element_text(size = 15)) + 
  scale_y_continuous(breaks = c(0, 256, 512, 768, 1024)) 
  
