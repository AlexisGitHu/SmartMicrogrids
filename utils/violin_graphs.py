import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

folder_path="results"
# List of your CSV files
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    df['File_Name'] = file.split(".")[0]  # Add a new column for file name
    dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Plot violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x="File_Name", y="List_of_rewards", data=combined_df)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.xlabel("Tipo de agente inteligente y alternativa")
plt.ylabel("Valor de la función de ganancia en 2 meses")
plt.title("Diagramas de violín por tipo de agente inteligente y alternativa")
plt.tight_layout()
plt.savefig("graficos/violin/violin_plot.png")