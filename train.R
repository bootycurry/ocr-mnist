source("hybrid_model.r")
source("cnn_model.r")

model <- hybrid_model

history <- model %>% fit(
  x = x_train, 
  y = y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.3,
  verbose = 1
)

plot(history)

results <- model %>% evaluate(x_test, y_test)
cat("Test loss:", results$loss, "\n")
cat("Test accuracy:", results$accuracy, "\n")

predictions <- model %>% predict(x_test)
predictions_classes <- apply(predictions, 1, which.max) - 1  
y_test_classes <- apply(y_test, 1, which.max) - 1 

# Confusion Matrix
conf_matrix <- table(Predicted = predictions_classes, Actual = y_test_classes)
print(conf_matrix)

#model %>% save_model("mnist_model.keras")