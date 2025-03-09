# Cargar librerías necesarias
library(readxl)
library(dplyr)
library(ggplot2)
library(pROC)
library(caret)
library(broom)
library(ineq)

# Cargar datos
df <- read_excel("Proyecto 1 Datos.xlsm", sheet = "E11")

# Separar conjuntos de entrenamiento y prueba
train <- df %>% filter(Sample == "Train")
test <- df %>% filter(Sample == "Test")

# Verificar valores nulos
print("Valores nulos en el dataset:")
print(colSums(is.na(df)))

# Definir variables objetivo
y_train <- as.numeric(train$Purchased)
y_test <- as.numeric(test$Purchased)


# Primer modelo: Con todas las variables 

# Definir variables predictoras
variables_modelo1 <- c("EstimatedSalary USD", "Gender", "Age")
X_train_modelo1 <- train[, variables_modelo1]
X_test_modelo1 <- test[, variables_modelo1]

# Convertir 'Gender' a valores numéricos
X_train_modelo1$Gender <- ifelse(X_train_modelo1$Gender == "Male", 1, 0)
X_test_modelo1$Gender <- ifelse(X_test_modelo1$Gender == "Male", 1, 0)

# Estandarización
scaler <- preProcess(X_train_modelo1, method = c("center", "scale"))
X_train_scaled_modelo1 <- predict(scaler, X_train_modelo1)
X_test_scaled_modelo1 <- predict(scaler, X_test_modelo1)

# Agregar constante para regresión logística
X_train_modelo1 <- cbind(Intercept = 1, X_train_scaled_modelo1)
X_test_modelo1 <- cbind(Intercept = 1, X_test_scaled_modelo1)

# Ajustar el modelo de regresión logística
modelo1 <- glm(Purchased ~ ., data = cbind(Purchased = y_train, X_train_modelo1), family = binomial)
print("Resumen del Modelo 1 (con Gender):")
print(summary(modelo1))

# Calcular probabilidades en el conjunto de prueba
test$Score_modelo1 <- predict(modelo1, newdata = X_test_modelo1, type = "link")
test$PD_modelo1 <- 1 / (1 + exp(-test$Score_modelo1))  # Probabilidad de compra

# Función para calcular el KS Score
ks_score <- function(y_real, y_pred) {
  roc_obj <- roc(y_real, y_pred)
  tpr <- roc_obj$sensitivities
  fpr <- 1 - roc_obj$specificities
  return(max(tpr - fpr))
}

# Validación del Modelo 1
ks_value_modelo1 <- ks_score(y_test, test$PD_modelo1)
gini_index_modelo1 <- 2 * auc(roc(y_test, test$PD_modelo1)) - 1
auc_value_modelo1 <- auc(roc(y_test, test$PD_modelo1))

cat("\nMétricas del Modelo 1 (con Gender):\n")
cat("KS Score:", ks_value_modelo1, "\n")
cat("Gini Index:", gini_index_modelo1, "\n")
cat("AUC Score:", auc_value_modelo1, "\n")


# Segundo modelo: Descartando Gender (solo EstimatedSalary USD y Age)


# Seleccionar variables (solo Edad y Salario, sin Gender)
variables_modelo2 <- c("EstimatedSalary USD", "Age")
X_train_modelo2 <- train[, variables_modelo2]
X_test_modelo2 <- test[, variables_modelo2]

# Estandarización de variables para el Modelo 2
scaler_modelo2 <- preProcess(X_train_modelo2, method = c("center", "scale"))
X_train_scaled_modelo2 <- predict(scaler_modelo2, X_train_modelo2)
X_test_scaled_modelo2 <- predict(scaler_modelo2, X_test_modelo2)

# Agregar constante
X_train_modelo2 <- cbind(Intercept = 1, X_train_scaled_modelo2)
X_test_modelo2 <- cbind(Intercept = 1, X_test_scaled_modelo2)

# Ajustar el modelo de regresión logística
modelo2 <- glm(Purchased ~ ., data = cbind(Purchased = y_train, X_train_modelo2), family = binomial)
print("\nResumen del Modelo 2 (sin Gender):")
print(summary(modelo2))

# Calcular probabilidades en el conjunto de prueba
test$Score_modelo2 <- predict(modelo2, newdata = X_test_modelo2, type = "link")
test$PD_modelo2 <- 1 / (1 + exp(-test$Score_modelo2))  # Probabilidad de compra

# Validación del Modelo 2
ks_value_modelo2 <- ks_score(y_test, test$PD_modelo2)
gini_index_modelo2 <- 2 * auc(roc(y_test, test$PD_modelo2)) - 1
auc_value_modelo2 <- auc(roc(y_test, test$PD_modelo2))

cat("\nMétricas del Modelo 2 (sin Gender):\n")
cat("KS Score:", ks_value_modelo2, "\n")
cat("Gini Index:", gini_index_modelo2, "\n")
cat("AUC Score:", auc_value_modelo2, "\n")

# ==================================================
# Gráficas y validaciones usando el Modelo 2 (mejor modelo)
# ==================================================

# Gráfica ROC
roc_obj_modelo2 <- roc(y_test, test$PD_modelo2)
plot(roc_obj_modelo2, main = "ROC Curve (Modelo 2)", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = paste("AUC =", round(auc_value_modelo2, 4)), col = "blue", lwd = 2)

# Gráfica KS Score
plot(roc_obj_modelo2$specificities, roc_obj_modelo2$sensitivities, type = "l", col = "blue",
     xlab = "False Positive Rate", ylab = "True Positive Rate", main = "Kolmogorov-Smirnov (KS) Score (Modelo 2)")
abline(a = 0, b = 1, lty = 2, col = "gray")
ks_index <- which.max(roc_obj_modelo2$sensitivities - roc_obj_modelo2$specificities)
abline(v = roc_obj_modelo2$specificities[ks_index], col = "red", lty = 2)
legend("bottomright", legend = paste("KS =", round(ks_value_modelo2, 4)), col = "red", lwd = 2)

# Gráfica Índice de Gini
lorenz_curve <- Lc(test$PD_modelo2)
plot(lorenz_curve, main = "Curva de Lorenz e Índice de Gini (Modelo 2)", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = paste("Gini =", round(gini_index_modelo2, 4)), col = "blue", lwd = 2)

# Distribución de Probabilidades de Compra
ggplot(test, aes(x = PD_modelo2, fill = as.factor(Purchased))) +
  geom_histogram(binwidth = 0.05, alpha = 0.6, position = "identity") +
  labs(title = "Distribución de Probabilidades (Modelo 2)", x = "Probabilidad de compra (PD)", y = "Frecuencia") +
  scale_fill_manual(values = c("red", "blue"), name = "Compradores", labels = c("No compradores", "Compradores"))

# Matriz de Confusión
predicted_classes_modelo2 <- ifelse(test$PD_modelo2 > 0.5, 1, 0)
conf_matrix_modelo2 <- confusionMatrix(as.factor(predicted_classes_modelo2), as.factor(y_test))
print("\nMatriz de Confusión (Modelo 2):")
print(conf_matrix_modelo2)

# Precisión del modelo
accuracy_modelo2 <- conf_matrix_modelo2$overall['Accuracy']
cat('Precisión del Modelo 2:', accuracy_modelo2, '\n')

# Crear un DataFrame con la información relevante
test$Predicted_Class_modelo2 <- predicted_classes_modelo2

# Dividir en compradores y no compradores
compradores_modelo2 <- test %>% filter(Predicted_Class_modelo2 == 1)
no_compradores_modelo2 <- test %>% filter(Predicted_Class_modelo2 == 0)

# Guardar en un archivo Excel
write.xlsx(list(Compradores = compradores_modelo2, No_compradores = no_compradores_modelo2),
           file = "clasificacion_compradores_modelo2.xlsx")

cat("\nArchivo Excel generado exitosamente: 'clasificacion_compradores_modelo2.xlsx'\n")