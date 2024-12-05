# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#

# Carga de librerias
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
import pickle
import numpy as np
import os
import pickle
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

# Se limpian los datos

def clean_df(path):
    
    d_frame=pd.read_csv(path,index_col=False,compression='zip')
    
    d_frame.rename(columns={'default payment next month': 'default'}, inplace=True)
    d_frame.drop(columns='ID', inplace=True)
    d_frame = d_frame.iloc[d_frame[(d_frame['EDUCATION'] != 0) & (d_frame['MARRIAGE'] != 0)].index]
    d_frame['EDUCATION'] = d_frame['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

 
    return d_frame

data_train = clean_df('files/input/train_data.csv.zip')
data_test = clean_df('files/input/test_data.csv.zip')

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train= data_train.drop(columns='default')
y_train = data_train['default']
x_test= data_test.drop(columns='default') 
y_test = data_test['default']


# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categóricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (Random Forest).



pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['EDUCATION', 'MARRIAGE', 'SEX'])
        ],
        remainder='passthrough'  # Mantiene las otras columnas sin cambios
    )),
    ('model', RandomForestClassifier(random_state=42))
])


# # Paso 4.
# # Optimice los hiperparametros del pipeline usando validación cruzada.
# # Use 10 splits para la validación cruzada. Use la función de precision
# # balanceada para medir la precisión del modelo.

#hiperparametros para gridsearch
#El param_grid es un diccionario que define los hiperparámetros a probar y sus valores correspondientes.
#La sintaxis model__ se refiere al nombre del paso en el pipeline donde se encuentra el modelo (model, definido previamente en el pipeline).
#n_estimator: Número de árboles en el bosque
#max_depth: Profundidad máxima de los árboles
#min_samples_split: Mínimo número de muestras necesarias para dividir un nodo.
#min_samples_leaf: Número mínimo de muestras en una hoja.
#max_features: Número máximo de características consideradas para dividir un nodo.

param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [None],
    'model__min_samples_split':[10],
    'model__min_samples_leaf': [4],
    'model__max_features': [25],
}

model= GridSearchCV(pipeline, 
                    param_grid, 
                    cv=10, 
                    scoring='balanced_accuracy',
                    n_jobs=-1,
                    verbose=2
)

model.fit(x_train, y_train)

#pipeline: El pipeline que incluye preprocesamiento y el modelo (preprocessor + RandomForestClassifier).
#param_grid: Diccionario de hiperparámetros definido anteriormente.
#cv=10:Define el número de particiones para la validación cruzada.
#scoring="balanced_accuracy":   Métrica utilizada para evaluar el modelo.
#verbose=1:Muestra mensajes detallados durante la ejecución


# # Paso 5.
# # Guarde el modelo como "files/models/model.pkl".


file_path = "files/models/model.pkl" #Especifica dónde se guardará el modelo serializado
os.makedirs(os.path.dirname(file_path), exist_ok=True)
#os.makedirs():Crea el directorio especificado (files/models/) y cualquier subdirectorio necesario.
#os.path.dirname(file_path):Extrae la ruta del directorio (files/models/) desde file_path.
#exist_ok=True:Evita errores si el directorio ya existe.

if os.path.exists(file_path):
    os.remove(file_path)

#os.path.exists(file_path):Comprueba si el archivo especificado ya existe.
#os.remove(file_path):Elimina el archivo si está presente.

# Guardar el modelo
with open(file_path, "wb") as file:
    pickle.dump(model, file)

#open(file_path, "wb"): Abre el archivo en modo escritura binaria (wb).
#pickle.dump(model, file):Serializa (convierte a formato binario) el objeto model y lo guarda en el archivo.

# #
# # Paso 6.
# # Calcule las metricas de precision, precision balanceada, recall,
# # y f1-score para los conjuntos de entrenamiento y prueba.
# # Guardelas en el archivo files/output/metrics.json. Cada fila
# # del archivo es un diccionario con las metricas de un modelo.
# # Este diccionario tiene un campo para indicar si es el conjunto
# # de entrenamiento o prueba. Por ejemplo:
# # {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# # {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}


with open('files/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


def metrics (y_true, y_pred, dataset):
    return {
    'type': 'metrics',
    'dataset': dataset,
    'precision': precision_score(y_true, y_pred),
    'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1_score': f1_score(y_true, y_pred)
    }

metrics_train = metrics(y_train, y_train_pred, 'train')
metrics_test = metrics(y_test, y_test_pred, 'test')


# # Paso 7.
# # Calcule las matrices de confusion para los conjuntos de entrenamiento y
# # prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# # del archivo es un diccionario con las metricas de un modelo.
# # de entrenamiento o prueba. Por ejemplo:
# #
# # {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# # {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
# #


output_dir = "files/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "metrics.json")
#output_dir: Especifica el directorio donde se guardará el archivo (files/output en este caso).
#os.makedirs(output_dir, exist_ok=True):Crea el directorio especificado, junto con cualquier subdirectorio necesario.
#Si el directorio ya existe, no genera un error debido a exist_ok=True.
#os.path.join(output_dir, "metrics.json"):Une el directorio output_dir con el nombre del archivo "metrics.json", creando la ruta completa del archivo.


# Eliminar el archivo si ya existe
if os.path.exists(output_path):
    os.remove(output_path)
    
# Crear las métricas de la matriz de confusión
def cm_matrix(cm, dataset):
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        'true_1': {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]}
    }

# Calcular las matrices de confusión
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

cm_matrix_train = cm_matrix(cm_train, 'train')
cm_matrix_test = cm_matrix(cm_test, 'test')

# Guardar las métricas
metrics = [metrics_train, metrics_test, cm_matrix_train, cm_matrix_test]
pd.DataFrame(metrics).to_json(output_path, orient='records', lines=True)