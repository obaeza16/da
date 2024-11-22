library(tidyverse)
setwd("~/eda_python")
data <- read_csv("./Malaga_Aparca/ocupappublicosmun.csv")
data
data <- data[1:11,]
data

cata <- read_csv("./Malaga_Aparca/catalogo.csv")
cata

cata <- cata %>% rename(ID = id)

# Clean up the dataframe
cleaned_df <- data %>%
  # Rename the single column for clarity
  rename(raw_data = `13:24:47 GMTETag: "b4-6276c30b3972b"`) %>%
  # Filter rows starting with "OCUPACION"
  filter(grepl("^OCUPACION,", raw_data)) %>%
  # Separate the data into columns
  separate(raw_data, into = c("Type", "ID", "Libres"), sep = ",") %>%
  # Drop the "Type" column (if not needed)
  select(-Type) %>%
  # Convert "Libres" to numeric
  mutate(Libres = as.numeric(Libres))

# Print the cleaned dataframe
print(cleaned_df)

final_df <- cleaned_df %>%
  left_join(cata, by = "ID")

final_df$capacidad <- c(rep(300, 10))
final_df$libres <- final_df$Libres
final_df <- select(final_df, -c(Libres))
write.csv(final_df,"./Malaga_Aparca/final_malaga_data.csv")



data_final <- read_csv("./Malaga_Aparca/final_malaga_data.csv", col_select = c(2:9))
# get and bind the new data_final
#rbind(values$df, get_new_data_final()) %>%
#filter(!is.na(x)) # filter the first value to prevent a first point in the middle of the plot

data_final <- head(data_final, 10)

data_final$capacidad <- as.numeric(data_final$capacidad)
data_final$logcapacidad <- log(data_final$capacidad)    
data_final