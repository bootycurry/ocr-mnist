library(keras3)

preprocess_data <- function(data, validation_split = 0.2) {
  x_train <- data$train$images
  y_train <- data$train$labels
  
  x_test <- data$test$images
  y_test <- data$test$labels
  
  x_train <- x_train / 255
  x_test <- x_test / 255
  
  # Reshape images to include channel dimension: (samples, height, width, channels)
  x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
  x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
  
  # One-hot encode labels
  y_train <- to_categorical(y_train, 10)
  y_test <- to_categorical(y_test, 10)
  
  return(list(
    x_train = x_train,
    y_train = y_train,
    x_test = x_test,
    y_test = y_test
  ))
}
