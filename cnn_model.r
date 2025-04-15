library(tensorflow)
library(keras3)

cnn_model <- keras_model_sequential()

cnn_model %>%
  layer_conv_2d(
    filters = 4, kernel_size = c(3, 3),
    padding = "same", input_shape = c(28, 28, 1)
  ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(
    filters = 4, kernel_size = c(3, 3)
  ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  
  layer_dense(4) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  layer_dense(10, activation = "softmax")

cnn_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

summary(cnn_model)