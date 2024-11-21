library(readr)
library(dplyr)
library(corrplot)
calidad_aire <- read_delim("eda_python/calidad-del-aire-datos-historicos-diarios.csv", 
                                                        delim = ";", escape_double = FALSE, trim_ws = TRUE)
View(calidad_aire)

# We represent some characteristics of the dataset
str(calidad_aire)
summary(calidad_aire)

# Generamos los histogramas de dos variables numéricas que presenta la tabla de
# datos: 03 (μg/m3) y PM10 ((μg/m3)
hist_O3 <- hist(calidad_aire$"O3 (ug/m3)", xlab = "O3 (ug/m3)", 
                ylab = "Frecuencia", xlim = c(0, 150), breaks = 1000)
hist_PM10 <- hist(calidad_aire$"PM10 (ug/m3)", xlab = "PM10 (ug/m3)",
                  ylab = "Frecuencia", xlim = c(0, 150), breaks = 1000)

# Ajustamos el tipo de la variable Fecha
calidad_aire$Fecha <- as.Date(calidad_aire$Fecha, format("%d/%m/%Y"))
# Ajustamos el tipo de la variable Provincia y Estación
unique(calidad_aire$Provincia)
calidad_aire$Provincia <- as.factor(calidad_aire$Provincia)
unique(calidad_aire$Estación)
calidad_aire$Estación <- as.factor(calidad_aire$Estación)

# Devuelve un vector lógico
is.na(calidad_aire)
# Devuelve un único valor lógico, cierto o falso, si existe algún valor ausente
any(is.na(calidad_aire))
# Devuelve el número de NAs que presenta la tabla
sum(is.na(calidad_aire))
# Devuelve el % de valores perdidos
mean(is.na(calidad_aire))

# Detección del número de valores perdidos en cada una de las columnas que 
# presenta la tabla
colSums(is.na(calidad_aire))
# Detección del % de valores perdidos en cada una de las columnas que
# presenta la tabla
colMeans(is.na(calidad_aire), round(2))

# Eliminación de las variables que presentan un % de NAs superior al 50%, para 
# ello se utiliza la función which() que permite realizar selecciones de datos 
# bajo alguna premisa.
calidad_aire <- calidad_aire[,-which(colMeans(is.na(calidad_aire)) >= 0.50)]

# Seleccionamos las variables numéricas que presenta la tabla iterando sobre 
# todas las columnas de la tabla mediante la función sapply()
columnas_numericas <- which(sapply(calidad_aire, is.numeric))
# Calculamos la media para cada una de las variables numéricas sin tener en 
# cuenta los NAs
cols_means <- colMeans(calidad_aire[, columnas_numericas], na.rm = TRUE)
# Sustituimos los valores NA por la media correspondiente a cada variable
for (x in columnas_numericas) {
calidad_aire[is.na(calidad_aire[,x]), x] <- round(cols_means[x],2)
}


histograma_O3 <- hist(calidad_aire$`O3 (ug/m3)`, main ="", xlab = "O3 (ug/m3)",
                      ylab = "Frecuencia", xlim = c(0, 150), breaks = 1000)

histograma_NO <- hist(calidad_aire$`NO (ug/m3)`, main ="", xlab = "NO (ug/m3)",
                      ylab = "Frecuencia", xlim = c(0, 150), breaks = 1000)

histograma_NO <- hist(calidad_aire$`NO2 (ug/m3)`, main ="", xlab = "NO2 (ug/m3)",
                      ylab = "Frecuencia", xlim = c(0, 150), breaks = 1000)


# Estadísticas necesarias para reproducir el gráfico de cajas y bigotes
boxplot.stats(calidad_aire$`NO2 (ug/m3)`)
outliers <- boxplot.stats(calidad_aire$`NO2 (ug/m3)`)
# Construcción del gráfico de cajas y bigotes
boxplot(calidad_aire$`O3 (ug/m3)`, horizontal = TRUE, xlab = "O3 (ug/m3)")
boxplot(calidad_aire$`NO2 (ug/m3)`, horizontal = TRUE, xlab = "NO2 (ug/m3)")

# Se genera una nueva tabla que no contiene los valores almacenados en el vector
# outliers$out, antes obtenido con la función boxplot.stats().
calidad_aire_NoOut <- calidad_aire[!(calidad_aire$`NO2 (ug/m3)` %in% outliers$out),]

# Construcción de los gráficos de cajas y bigotes
boxplot(calidad_aire$`NO2 (ug/m3)`, xlab = "NO2 (ug/m3)")
boxplot(calidad_aire_NoOut$`NO2 (ug/m3)`, xlab = "NO2 (ug/m3)")

# Número de categorías que presenta la variable Provincia
count(calidad_aire, "Provincia")
# Construcción del gráfico de barras para la variable Provincia
ggplot_provincias <- ggplot(calidad_aire)+ geom_bar(aes(x = Provincia, 
                                                        fill = Provincia)) + 
  xlab("Provincias") + ylab("Nºobservaciones") +theme(axis.text.x = 
                                                        element_text(angle = 30))
ggplot_provincias

# Eliminamos las filas que pertenecen al factor "Madrid"
eliminar_Madrid <- calidad_aire$Provincia %in% c("Madrid")
calidad_aire_SM <- calidad_aire[!eliminar_Madrid,]
# Eliminamos el factor "Madrid"
calidad_aire_SM$Provincia <- as.factor(calidad_aire_SM$Provincia)
calidad_aire_SM$Provincia <- droplevels(calidad_aire_SM$Provincia)
# Con la función levels() verificamos la eliminación de la categoría 
#   "Madrid" de la variable
levels(calidad_aire_SM$Provincia)

# Seleccionamos las variables numéricas situadas en las columnas 2 a 6 
# de la tabla de calidad del aire (NO, NO2, O3, PM10, PM25 y SO2)
num_variables <- calidad_aire[,c(2,3,4,5,6)]
# Calculamos la matriz de coeficientes de correlación entre las variables numéricas
correlacion <- cor(num_variables)
# Gráfico de correlaciones indicando la forma en la que se representa la 
# correlación (un cuadrado que varía en tamaño según la fortaleza). Para la 
# generación de este gráfico es necesario instalar y cargar la librería corrplot
corrplot(correlacion, method = “square”)