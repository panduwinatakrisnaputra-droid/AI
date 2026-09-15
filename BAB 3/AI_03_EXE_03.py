import pandas as pd
from matplotlib import pyplot as plt

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df = pd.read_csv(url)

# Encode species names into numerical values for coloring points
species_map = {species: i for i, species in enumerate(df['species'].unique())}
colors = df['species'].map(species_map)

# Scatter plot of Sepal Length vs Sepal Width
plt.figure(figsize=(8, 6))
plt.scatter(df['sepal_length'], df['sepal_width'], c=colors, cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Sepal Length vs Sepal Width')
plt.grid(True)
plt.show()