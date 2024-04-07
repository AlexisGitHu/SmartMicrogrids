import matplotlib.pyplot as plt
import os

folder_path="results"
# Define your specific names for the X-axis
names = [file.split(".")[0] for file in os.listdir(folder_path) if file.endswith('.csv')]

print(names)

# Define corresponding values for the Y-axis
# values = [0.0341, 0.0313, 0.0354, 0.0392, 0.0257,0.0419,0.0371]
values=[0.0392,0.0354,0.0419,0.0257,0.0313,0.0371,0.0341]
# Create a scatter plot
plt.scatter(names, values, color='blue')

# Add labels and title
plt.xlabel('Tipo de agente inteligente y alternativa')
plt.ylabel('Valor en media de ahorro en 2 meses')
plt.title('Medias de ahorro por tipos en 1000 simulaciones')

# Display the plot
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()
plt.savefig("graficos/puntos/resultado_final.png")