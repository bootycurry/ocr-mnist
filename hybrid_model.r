library(tensorflow)
library(keras3)

hybrid_model <- keras_model_sequential()


hybrid_model %>%
  layer_conv_2d(
    filters = 32, kernel_size = c(3, 3),
    padding = "same", input_shape = c(28, 28, 1)
  ) %>%
  layer_activation_leaky_relu(0.1) %>%
  
  layer_conv_2d(
    filters = 64, kernel_size = c(3, 3)
  ) %>%
  layer_activation_leaky_relu(0.1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same") %>%
  layer_activation_leaky_relu(0.1) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%

  layer_reshape(target_shape = c(-1, 64)) %>% 

  # LSTM layers
  layer_lstm(units = 128, return_sequences = TRUE) %>%
  layer_dropout(0.25) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.25) %>%

  layer_dense(256) %>%
  layer_activation_leaky_relu(0.1) %>%
  layer_dropout(0.5) %>%
  layer_dense(10, activation = "softmax")

hybrid_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

summary(hybrid_model)

