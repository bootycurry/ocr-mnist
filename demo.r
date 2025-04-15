library(keras3)
library(tensorflow)
library(ggplot2)
library(magick)
library(grid)
library(png)

# Function to preprocess an input image to match the model's expected format
preprocess_image <- function(img_path) {
  # Read image and convert to grayscale
  img <- image_read(img_path)
  img <- image_convert(img, colorspace = "gray")
  
  # Resize to 28x28 pixels
  img <- image_resize(img, "28x28!")
  
  # Convert to array and normalize - with explicit type conversion
  img_array <- as.numeric(image_data(img))
  dim(img_array) <- c(28, 28, 1)  # Reshape using dim assignment
  img_array <- array(img_array, dim = c(1, 28, 28, 1))  # Create a new array with explicit dimensions
  img_array <- img_array / 255.0
  
  return(img_array)
}

# Function to predict digit from image and show probabilities
predict_digit <- function(img_path, model_path = "mnist_model") {
  # Load the saved model - try both with and without .keras extension
  tryCatch({
    model <- load_model(model_path)
  }, error = function(e) {
    tryCatch({
      model <- load_model(paste0(model_path, ".keras"))
    }, error = function(e2) {
      stop("Could not load model. Try providing the full path to the model.")
    })
  })
  
  # Preprocess the image
  img_array <- preprocess_image(img_path)
  
  # Original image for display
  original_img <- image_read(img_path)
  
  # Get predictions
  predictions <- model %>% predict(img_array)
  predicted_class <- which.max(predictions) - 1  # Adjust for 0-based indexing
  
  # Create data frame for plotting probabilities
  prob_df <- data.frame(
    Digit = 0:9,
    Probability = as.numeric(predictions[1,])
  )
  
  # Create probability plot
  prob_plot <- ggplot(prob_df, aes(x = factor(Digit), y = Probability)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_bar(stat = "identity", 
             data = prob_df[prob_df$Digit == predicted_class,], 
             fill = "darkgreen") +
    labs(title = paste("Predicted Digit:", predicted_class),
         subtitle = paste("Confidence:", round(max(predictions) * 100, 2), "%"),
         x = "Digit",
         y = "Probability") +
    theme_minimal() +
    ylim(0, 1)
  
  # Convert image to raster for plotting
  img_raster <- image_ggplot(original_img)
  
  # Display results
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1, 2)))
  print(img_raster, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(prob_plot, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
  
  # Return prediction results
  return(list(
    predicted_digit = predicted_class,
    confidence = max(predictions) * 100,
    probabilities = as.numeric(predictions[1,])
  ))
}
  result <- predict_digit("test/num_5.png")
  print(result)

