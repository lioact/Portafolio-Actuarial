#%%
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_excel("Proyecto 1 Datos.xlsm", sheet_name="E11")

# Separar conjuntos de entrenamiento y prueba
train = df[df['Sample'] == 'Train'].copy()
test = df[df['Sample'] == 'Test'].copy()

# Verificar valores nulos
print("Valores nulos en el dataset:")
print(df.isnull().sum())

# Definir variables objetivo
y_train = train['Purchased'].astype(int)
y_test = test['Purchased'].astype(int)

# ==================================================
# Primer modelo: Con todas las variables (Gender incluido)
# ==================================================

# Definir variables predictoras
variables_modelo1 = ['EstimatedSalary USD', 'Gender', 'Age']
X_train_modelo1 = train[variables_modelo1]
X_test_modelo1 = test[variables_modelo1]

# Convertir 'Gender' a valores numéricos
X_train_modelo1['Gender'] = X_train_modelo1['Gender'].map({'Male': 1, 'Female': 0})
X_test_modelo1['Gender'] = X_test_modelo1['Gender'].map({'Male': 1, 'Female': 0})

# Estandarización
scaler = StandardScaler()
X_train_scaled_modelo1 = scaler.fit_transform(X_train_modelo1)
X_test_scaled_modelo1 = scaler.transform(X_test_modelo1)

# Convertir a DataFrame
X_train_modelo1 = pd.DataFrame(X_train_scaled_modelo1, columns=variables_modelo1, index=X_train_modelo1.index)
X_test_modelo1 = pd.DataFrame(X_test_scaled_modelo1, columns=variables_modelo1, index=X_test_modelo1.index)

# Agregar constante para regresión logística
X_train_modelo1 = sm.add_constant(X_train_modelo1)
X_test_modelo1 = sm.add_constant(X_test_modelo1)

# Ajustar el modelo de regresión logística
modelo1 = sm.Logit(y_train, X_train_modelo1).fit()

# Mostrar el resumen del modelo
print("Resumen del Modelo 1 (con Gender):")
print(modelo1.summary())

# Calcular probabilidades en el conjunto de prueba
test["Score_modelo1"] = modelo1.params[0] + modelo1.params[1] * X_test_modelo1["EstimatedSalary USD"] + modelo1.params[2] * X_test_modelo1["Gender"] + modelo1.params[3] * X_test_modelo1["Age"]
test["PD_modelo1"] = 1 / (1 + np.exp(-test["Score_modelo1"]))  # Probabilidad de compra

# Función para calcular el KS Score
def ks_score(y_real, y_pred):
    fpr_ks, tpr_ks, _ = roc_curve(y_real, y_pred)  # Cambiamos nombres para evitar sombreado
    return max(tpr_ks - fpr_ks)

# Validación del Modelo 1
ks_value_modelo1 = ks_score(y_test, test["PD_modelo1"])
gini_index_modelo1 = 2 * roc_auc_score(y_test, test["PD_modelo1"]) - 1
auc_value_modelo1 = roc_auc_score(y_test, test["PD_modelo1"])

print("\nMétricas del Modelo 1 (con Gender):")
print(f"KS Score: {ks_value_modelo1:.4f}")
print(f"Gini Index: {gini_index_modelo1:.4f}")
print(f"AUC Score: {auc_value_modelo1:.4f}")

# ==================================================
# Segundo modelo: Descartar Gender (solo EstimatedSalary USD y Age)
# ==================================================

# Seleccionar variables (solo Edad y Salario, sin Gender)
variables_modelo2 = ['EstimatedSalary USD', 'Age']
X_train_modelo2 = train[variables_modelo2]
X_test_modelo2 = test[variables_modelo2]

# Estandarización de variables
X_train_scaled_modelo2 = scaler.fit_transform(X_train_modelo2)
X_test_scaled_modelo2 = scaler.transform(X_test_modelo2)

# Convertir a DataFrame
X_train_modelo2 = pd.DataFrame(X_train_scaled_modelo2, columns=variables_modelo2, index=X_train_modelo2.index)
X_test_modelo2 = pd.DataFrame(X_test_scaled_modelo2, columns=variables_modelo2, index=X_test_modelo2.index)

# Agregar constante
X_train_modelo2 = sm.add_constant(X_train_modelo2)
X_test_modelo2 = sm.add_constant(X_test_modelo2)

# Ajustar el modelo de regresión logística
modelo2 = sm.Logit(y_train, X_train_modelo2).fit()

# Mostrar el resumen del modelo
print("\nResumen del Modelo 2 (sin Gender):")
print(modelo2.summary())

# Calcular probabilidades en el conjunto de prueba
test["Score_modelo2"] = modelo2.params[0] + modelo2.params[1] * X_test_modelo2["EstimatedSalary USD"] + modelo2.params[2] * X_test_modelo2["Age"]
test["PD_modelo2"] = 1 / (1 + np.exp(-test["Score_modelo2"]))  # Probabilidad de compra

# Validación del Modelo 2
ks_value_modelo2 = ks_score(y_test, test["PD_modelo2"])
gini_index_modelo2 = 2 * roc_auc_score(y_test, test["PD_modelo2"]) - 1
auc_value_modelo2 = roc_auc_score(y_test, test["PD_modelo2"])

print("\nMétricas del Modelo 2 (sin Gender):")
print(f"KS Score: {ks_value_modelo2:.4f}")
print(f"Gini Index: {gini_index_modelo2:.4f}")
print(f"AUC Score: {auc_value_modelo2:.4f}")

# ==================================================
# Gráficas y validaciones usando el Modelo 2 (mejor modelo)
# ==================================================

# Gráfica ROC
fpr, tpr, _ = roc_curve(y_test, test["PD_modelo2"])
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_value_modelo2:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Modelo 2)')
plt.legend()
plt.show()

# Gráfica KS Score
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label='Curva ROC', color='blue')
plt.plot(fpr, fpr, linestyle='--', color='gray', label='Línea de referencia')
plt.axvline(x=fpr[np.argmax(tpr - fpr)], color='red', linestyle='--', label=f'KS = {ks_value_modelo2:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kolmogorov-Smirnov (KS) Score (Modelo 2)')
plt.legend()
plt.show()

# Gráfica Índice de Gini
sorted_proba = np.sort(test["PD_modelo2"])
cumulative_actual = np.linspace(0, 1, len(sorted_proba))
plt.figure(figsize=(6, 6))
plt.plot(cumulative_actual, sorted_proba, label=f'Curva de Lorenz (Gini = {gini_index_modelo2:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Línea de igualdad')
plt.xlabel('Población acumulada')
plt.ylabel('Tasa de compradores acumulada')
plt.title('Curva de Lorenz e Índice de Gini (Modelo 2)')
plt.legend()
plt.show()

# Distribución de Probabilidades de Compra
plt.figure(figsize=(6, 5))
sns.histplot(test[test["Purchased"] == 1]["PD_modelo2"], bins=20, color='blue', label='Compradores', kde=True)
sns.histplot(test[test["Purchased"] == 0]["PD_modelo2"], bins=20, color='red', label='No compradores', kde=True)
plt.xlabel("Probabilidad de compra (PD)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Probabilidades (Modelo 2)")
plt.legend()
plt.show()

# Matriz de Confusión
predicted_classes_modelo2 = (test["PD_modelo2"] > 0.5).astype(int)
conf_matrix_modelo2 = confusion_matrix(y_test, predicted_classes_modelo2)
conf_matrix_df_modelo2 = pd.DataFrame(conf_matrix_modelo2, index=["Real 0 (No Compra)", "Real 1 (Compra)"],
                              columns=["Predicho 0 (No Compra)", "Predicho 1 (Compra)"])
print("\nMatriz de Confusión (Modelo 2):\n", conf_matrix_df_modelo2)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_df_modelo2, annot=True, fmt="d", cmap="Blues", linewidths=1, square=True)
plt.xlabel("Clase Predicha")
plt.ylabel("Clase Real")
plt.title("Matriz de Confusión (Modelo 2)")
plt.show()

# Precisión del modelo
accuracy_modelo2 = accuracy_score(y_test, predicted_classes_modelo2)
print(f'Precisión del Modelo 2: {accuracy_modelo2:.2f}')

# Crear un DataFrame con la información relevante
test["Predicted_Class_modelo2"] = predicted_classes_modelo2  # Agregar la clasificación predicha

# Dividir en compradores y no compradores
compradores_modelo2 = test[test["Predicted_Class_modelo2"] == 1]
no_compradores_modelo2 = test[test["Predicted_Class_modelo2"] == 0]

# Guardar en un archivo Excel
with pd.ExcelWriter("clasificacion_compradores_modelo2.xlsx") as writer:
    compradores_modelo2.to_excel(writer, sheet_name="Compradores", index=False)
    no_compradores_modelo2.to_excel(writer, sheet_name="No compradores", index=False)

print("\nArchivo Excel generado exitosamente: 'clasificacion_compradores_modelo2.xlsx'")