library(dslabs)
df <- dslabs::read_mnist(path = "./data")

display_mnist_example <- function(data, index = 1) {
  image_vector <- data$train$images[index, ]
  #print(paste("length of one row:",length(image_vector)))
  label <- data$train$labels[index]
  
  image_matrix <- matrix(image_vector, nrow = 28, byrow = TRUE)
  #print(image_matrix)
  par(mar = c(0.5, 0.5, 2, 0.5))
  image(t(apply(image_matrix, 2, rev)), 
        col = grey.colors(256, start = 1, end = 0),
        axes = FALSE, 
        main = paste("Label:", label))
}


display_mnist_example(df, index = 100)


data_preprocessed <- preprocess_data(df)

x_train <- data_preprocessed$x_train
y_train <- data_preprocessed$y_train
x_test <- data_preprocessed$x_test
y_test <- data_preprocessed$y_test