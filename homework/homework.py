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
import pandas as pd
import numpy as np
import os
import json
import gzip
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# Funcion para limpiar los datos

def clean_data(path):
    
    df=pd.read_csv(
        path,
        index_col=False,
        compression='zip')
    
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns='ID', inplace=True)
    df = df.iloc[df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)].index]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

 
    return df

df_train = clean_data('files/input/train_data.csv.zip')
df_test = clean_data('files/input/test_data.csv.zip')

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train= df_train.drop(columns='default')
y_train = df_train['default']
x_test= df_test.drop(columns='default')
y_test = df_test['default']

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).

steps=[('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['EDUCATION', 'MARRIAGE', 'SEX'])  # Creación de variables ficticticias (caracteristicas binarias)
        ],
        remainder='passthrough'  # Mantener el resto de las columnas sin cambios
    )),
       (('Random_Forest_Classifier', RandomForestClassifier(random_state=42)))
    ]

pipeline = Pipeline(steps)


# # Paso 4.
# # Optimice los hiperparametros del pipeline usando validación cruzada.
# # Use 10 splits para la validación cruzada. Use la función de precision
# # balanceada para medir la precisión del modelo.


param_grid = {
    'Random_Forest_Classifier__n_estimators': [100],
    'Random_Forest_Classifier__max_depth': [None],
    'Random_Forest_Classifier__min_samples_split':[10],
    'Random_Forest_Classifier__min_samples_leaf': [4],
    'Random_Forest_Classifier__max_features': [25], 
}
#Las claves son el nombre del paso del pipeline seguido de un doble guión bajo, seguido del nombre del hiperparámetro.

#realizamos una búsqueda en cuadrícula sobre nuestros parámetros instanciando el objeto GridSearchCV.
model= GridSearchCV(pipeline, 
                    param_grid, 
                    cv=10, 
                    scoring='balanced_accuracy',
                    n_jobs=-1,
            
)

model.fit(x_train, y_train)


# Paso 5: Guardar el modelo

models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)

#os.makedirs: Crea un directorio en la ruta especificada, incluidos todos los directorios intermedios necesarios. 
# Por ejemplo, si files no existe, lo crea junto con models.
# exist_ok=True: Evita que se genere un error si el directorio ya existe.
# Si el directorio existe, simplemente no realiza ninguna acción.

#guardar el modelo
model_path = "files/models/model.pkl.gz"

with gzip.open(model_path, "wb") as file:
    pickle.dump(model, file)
    
#Abre el archivo en modo escritura binaria ("wb") y lo comprime utilizando el formato gzip.
#Serializa el objeto model (generalmente, un modelo entrenado) y lo guarda en el archivo comprimido.

# Paso 6: Cálculo de métricas
def metrics_calculate(model, x_train, x_test, y_train, y_test):
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        }
    ]

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')
            
#Abre (o crea) el archivo metrics.json en el directorio files/output en modo escritura ("w").
#El uso de with asegura que el archivo se cierre correctamente después de escribir.


# Paso 7: Cálculo de matrices de confusión
def calculate_confusion_matrices(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    matrices = [
        {
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }
    ]

    with open("files/output/metrics.json", "a") as f:
        for matrix in matrices:
            f.write(json.dumps(matrix) + '\n')

# Ejecutar cálculo de métricas y matrices
metrics_calculate(model, x_train, x_test, y_train, y_test)
calculate_confusion_matrices(model, x_train, x_test, y_train, y_test)