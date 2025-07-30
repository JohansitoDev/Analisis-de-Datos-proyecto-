import pandas as pd
from sklearn.datasets import load_iris
import os

def load_iris_data(file_path='data/iris.csv'):
    if not os.path.exists(file_path):
        print("Descargando el dataset Iris...")
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        df.to_csv(file_path, index=False)
        print("Dataset Iris guardado en 'data/iris.csv'")
    else:
        print("Cargando dataset Iris desde 'data/iris.csv'")
    
    df = pd.read_csv(file_path)
    return df

def basic_data_info(df):
    print("\n--- Información Básica del Dataset ---")
    print(df.info())
    print("\n--- Primeras 5 Filas ---")
    print(df.head())
    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe())
    print("\n--- Conteo de Valores de Especies ---")
    print(df['species'].value_counts())

def check_missing_values(df):
    print("\n--- Valores Nulos por Columna ---")
    print(df.isnull().sum())
    
def preprocess_data(df):
 
    print("\n--- Datos preprocesados (si aplica) ---")
    return df