#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

ui <- fluidPage(
  titlePanel("Adjacency Matrix Plot"),
  sidebarLayout(
    sidebarPanel(
      numericInput("c_value", "Value of c:", 
                   min = 0,
                   max = 9,
                   value = 0),
      checkboxInput("conservative_value", "Conservative:", value = TRUE),
      checkboxInput("censored_value", "Censored:", value = TRUE),
      actionButton("update_plot", "Update Plot")
    ),
    mainPanel(
      plotOutput("matrix_plot")
    )
  )
)

server <- function(input, output) {
  observeEvent(input$update_plot, {
    output$matrix_plot <- renderPlot({
      X <- filter_matrices(c = input$c_value,
                           censored = input$censored_value,
                           conservative = input$conservative_value)
      
      # Handle cases where no matrices are selected
      if (length(unlist(X)) == 0) {
        return(ggplot() + annotate("text", x = 0.5, y = 0.5, label = "No matrices selected with these criteria."))
      }
      
      # Calculate the average matrix
      y <- as.matrix(Reduce("+", lapply(X, function(x) Reduce("+", x))) / (length(unlist(X))/length(X[[1]][[1]])))
      
      melted_data <- melt(y)
      
      ggplot(melted_data, aes(x = Var1, y = Var2, fill = value)) +
        geom_tile(color = "black") +
        geom_text(aes(label = round(value, 2)), color = "white", size = 4) +
        coord_fixed() +
        labs(x = "Indicators", y = "Indicators", fill = "Value") +
        theme_minimal()
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
