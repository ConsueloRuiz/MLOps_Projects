import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 1. Cargar los datos (ejemplo con datos ficticios o tus propios datos)
# Es necesario tener dos DataFrames de pandas: uno de referencia y uno actual.
# Puedes cargar tus datos desde CSV, por ejemplo:
# reference_data = pd.read_csv('ruta/a/datos_referencia.csv')
# current_data = pd.read_csv('ruta/a/datos_actuales.csv')

# Usaremos datos de ejemplo para ilustrar:
from evidently.datasets.base import create_fake_dataset
reference_data = create_fake_dataset(size=100)
current_data = create_fake_dataset(size=100, n_features=5, dataset_type='regression')


# 2. Definir el tipo de reporte a generar
# En este caso, usamos un preset para Data Drift.
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])


# 3. Ejecutar el reporte comparando los datos
# El primer argumento es el conjunto de datos actual, el segundo es el de referencia.
data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None)


# 4. Mostrar el reporte o guardarlo

# Para mostrarlo directamente en un entorno como Jupyter Notebook o Google Colab:
# data_drift_report

# Para guardar el reporte como un archivo HTML independiente:
data_drift_report.save_html("reporte_evidently_datadrift.html")

print("Reporte Evidently generado y guardado como 'reporte_evidently_datadrift.html'")
