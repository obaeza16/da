# Extracted from https://datos.gob.es/sites/default/files/doc/file/guia_eda_python.pdf
# Import initial libraries
import pandas as pd
import os

# Histograms
import matplotlib.pyplot as plt
import numpy as np

# Load data to DataFrame
# Data downloaded from https://datosabiertos.jcyl.es/web/jcyl/risp/es/medio-ambiente/calidad_aire_historico/1284212629698.csv
calidad_aire = pd.read_csv("./calidad-del-aire-datos-historicos-diarios.csv", sep=";")

# Head of dataframe
calidad_aire.head(5)

# Information of dataframe
print(calidad_aire.info())

# Summary analysis
print(calidad_aire.describe())

# Generating histograms for all numeric variables
num_col = calidad_aire.select_dtypes(include=[np.number]).columns
# Calculate number of rows and columns
n = len(num_col)
nrows = 3
ncols = min(n, 3)

# Create figure and subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
fig.suptitle("Distribución de Variables Numéricas", fontsize=16)

# Flatten array
axes = axes.flatten() if n > 3 else [axes]

# Crear histogramas para cada variable numérica
for i, col in enumerate(num_col):
    ax = axes[i]
    calidad_aire[col].hist(ax=ax, bins=50, edgecolor="black")
    ax.set_title(f"Distribución de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")

# Hide empty plots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Transformation of data
# Adjust Fecha data type
calidad_aire['Fecha'] = pd.to_datetime(calidad_aire['Fecha'], errors='coerce')
# Adjust Provincia data type
print(calidad_aire['Provincia'].unique())
calidad_aire['Provincia'] = calidad_aire['Provincia'].astype('category')
# Adjust Estación data type
print(calidad_aire['Estación'].unique())
calidad_aire['Estación'] = calidad_aire['Estación'].astype('category')

# Working with NAs
# Devuelve un DataFrame booleano
calidad_aire.isna()
# Devuelve True si hay al menos un valor ausente
calidad_aire.isna().any().any()
# Devuelve el número total de NaN que presenta el DataFrame
print(calidad_aire.isna().sum().sum())
# Devuelve el % de valores perdidos
print(calidad_aire.isna().mean().mean())
# Detección del número de valores perdidos en cada una de las columnas
calidad_aire.isna().sum()
# Detección del % de valores perdidos en cada una de las columnas
calidad_aire.isna().mean().round(2)

# Make copy of original dataset
calidad_aire_original = calidad_aire.copy()
# Eliminación de las variables que presentan un % de NaN superior al 50%
calidad_aire = calidad_aire.loc[:, calidad_aire.isna().mean() < 0.5]
print(f" Tras esta operación, contamos con {len(calidad_aire.columns)} columnas")

# We will fill the rest of NAs by the mean of the columns
# Seleccionamos las variables numéricas
columnas_numericas = calidad_aire.select_dtypes(include=[np.number]).columns
# Calculamos la media para cada una de las variables numéricas sin tener en cuenta los NaN
cols_mean = calidad_aire[columnas_numericas].mean()
# Sustituimos los valores NaN por la media correspondiente a cada variable
calidad_aire[columnas_numericas] = calidad_aire[columnas_numericas].fillna(cols_mean)

# Watching for outliers

plt.hist(calidad_aire['O3 (ug/m3)'], bins=100, range=(0, 150), color='blue', edgecolor='black')
plt.title('Distribución de O3 (ug/m3)')
plt.xlabel('O3 (ug/m3)')
plt.ylabel('Frecuencia')
plt.xlim(0,150)
plt.tight_layout()
plt.show()
# We can see a huge outlier in this value, we should remove

# We will represent the boxplots
import seaborn as sns
# Estadísticas necesarias para reproducir el gráfico de cajas y bigotes
Q1 = calidad_aire['O3 (ug/m3)'].quantile(0.25)
Q3 = calidad_aire['O3 (ug/m3)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Estadísticas para O3:")
print(f"Q1 - 1.5IQR = {lower_bound:.2f}")
print(f"Q1 = {Q1:.2f}")
print(f"Mediana = {calidad_aire['O3 (ug/m3)'].median():.2f}")
print(f"Q3 = {Q3:.2f}")
print(f"Q3 + 1.5IQR = {upper_bound:.2f}")
print(f"Número de observaciones: {len(calidad_aire['O3 (ug/m3)'])}")
print(f"Número de outliers: {sum((calidad_aire['O3 (ug/m3)'] < lower_bound) | (calidad_aire['O3 (ug/m3)'] > upper_bound))}")
# Construcción del gráfico de cajas y bigotes
plt.figure(figsize=(10, 6))
sns.boxplot(x=calidad_aire['O3 (ug/m3)'])
plt.title('Gráfico de cajas y bigotes para O3 (μg/m3)')
plt.xlabel('O3 (μg/m3)')
plt.show()

# Número de categorías que presenta la variable Provincia
categoria_counts = calidad_aire['Provincia'].value_counts()
# Construcción del gráfico de barras para la variable Provincia
plt.figure(figsize=(10, 6))
sns.barplot(x=categoria_counts.index, y=categoria_counts.values, palette='Blues')
plt.xlabel('Provincias')
plt.ylabel('Nº observaciones')
plt.xticks(rotation=30)
plt.title('Distribución de la variable Provincia')
plt.show()

# We will now delete atipical values

# Se genera una nueva tabla que no contiene los valores identificados como atípicos
calidad_aire_NoOut = calidad_aire[(calidad_aire['O3 (ug/m3)'] >= lower_bound) &
(calidad_aire['O3 (ug/m3)'] <= upper_bound)]
# Construcción de los gráficos de cajas y bigotes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(x=calidad_aire['O3 (ug/m3)'], ax=ax1)
ax1.set_title('O3 (μg/m3) con outliers')
ax1.set_xlabel('O3 (μg/m3)')
sns.boxplot(x=calidad_aire_NoOut['O3 (ug/m3)'], ax=ax2)
ax2.set_title('O3 (μg/m3) sin outliers')
ax2.set_xlabel('O3 (μg/m3)')
plt.tight_layout()
plt.show()

# Eliminamos las filas que pertenecen al factor “Madrid”
calidad_aire_SM = calidad_aire[calidad_aire['Provincia'] != 'Madrid'].copy()
# Eliminamos el factor “Madrid”
calidad_aire_SM['Provincia'] = calidad_aire_SM['Provincia'].astype('category').cat.remove_unused_categories()
# Verificamos la eliminación de la categoría "Madrid"
print(calidad_aire_SM['Provincia'].cat.categories)


num_variables = calidad_aire.select_dtypes(include=[np.number])
# Calculamos la matriz de coeficientes de correlación entre las variables numéricas
correlacion = num_variables.corr()
# Configuración del gráfico de correlación
plt.figure(figsize=(10, 8))
# Gráfico de correlaciones utilizando un mapa de calor
sns.heatmap(correlacion, annot=True, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Matriz de correlaciones entre variables')
plt.show()

# Now we will use YData Profiling, a tool to create automated EDA 
from ydata_profiling import ProfileReport
report = ProfileReport(calidad_aire_original, title='EDA automático')
report_file = 'reporte_calidad_aire.html'
report.to_file(report_file)

