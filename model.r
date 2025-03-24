library(tensorflow)
library(keras3)

model <- keras_model_sequential()


model %>%
  layer_conv_2d(
    filters = 16, kernel_size = c(3, 3),
    padding = "same", input_shape = c(28, 28, 1)
  ) %>%
  layer_activation_leaky_relu(0.1) %>%
  layer_conv_2d(
    filters = 32, kernel_size = c(3, 3)
  ) %>%
  layer_activation_leaky_relu(0.1) %>%
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  # 1 additional hidden 2D convolutional layers
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same") %>%
  layer_activation_leaky_relu(0.1) %>%
  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  # Reshape for LSTM - treating rows as time steps
  # The CNN output shape will be batch_size x height x width x channels
  # We need to reshape to batch_size x time_steps x features
  layer_reshape(target_shape = c(-1, 64)) %>% # -1 will be inferred from the input

  # LSTM layers
  layer_lstm(units = 128, return_sequences = TRUE) %>%
  layer_dropout(0.25) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.25) %>%

  layer_dense(256) %>%
  layer_activation_leaky_relu(0.1) %>%
  layer_dropout(0.5) %>%
  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

# Print model summary
summary(model)
