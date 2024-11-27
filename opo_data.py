import pandas as pd
import json

with open("./packt/resultat_fase_oposicio.json", "r") as file:
    data = json.load(file)
data

with open("./packt/resultat_fase_concurs.json", "r") as file:
    data_con = json.load(file)
data_con

# Normalize the JSON data
df = pd.json_normalize(data, record_path="Resultats")
df_con = pd.json_normalize(data_con, record_path="Resultats")

# Display the DataFrame
print(df)
df.info()
print(df_con)
df_con.info()

# Change type of numerical columns

df["Resultat Teorica"] = df["Resultat Teorica"].str.replace(",", ".").astype(float)
df["Resultat prova Practica"] = (
    df["Resultat prova Practica"].str.replace(",", ".").astype(float)
)
df["Total fase oposicio"] = (
    df["Total fase oposicio"].str.replace(",", ".").astype(float)
)
# We convert all dataframe columns but the first to numeric
df_con.set_index("Registre", inplace=True)
df_con = df_con.apply(pd.to_numeric, errors='coerce')

df.info(memory_usage='deep')
df.drop(columns="A", inplace=True)
df.set_index("Registre", inplace=True)
df
df.info(memory_usage='deep')

df_con
df_con.info(memory_usage='deep')

# We join the two dataframes by registre
df_tot = pd.DataFrame.join(self= df, other= df_con, on='Registre')
df_tot


df_tot['resultat_final'] = df_tot['Total fase oposicio'] + df_tot['Total concurs']
df_tot = df_tot.sort_values(by='resultat_final',ascending=False)
df_tot