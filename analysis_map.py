import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

plt.style.use("seaborn-v0_8")

# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet("embeddings_openai.parquet")
print(df)

df = df[df['country_code'] == "CH"]
embeddings = np.array(df['concat_embeddings'].tolist())

pca = PCA(n_components=1)
reduced_embeddings = pca.fit_transform(embeddings)

# Extracting the first PCA component for coloring
pca_component1 = reduced_embeddings[:, 0]

# Plotting
fig, ax = plt.subplots(figsize=(15, 10))
# https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
margin = 2
map = Basemap(ax=ax, llcrnrlon=min(df['longitude']) - margin,llcrnrlat=min(df['latitude']) - margin,urcrnrlon=max(df['longitude']) + margin,urcrnrlat=max(df['latitude']) + margin)

map.drawcoastlines()
map.drawcountries()

norm = plt.Normalize(vmin=pca_component1.min(), vmax=pca_component1.max())
colors = plt.cm.jet(norm(pca_component1))  # Normalized colors

# Scatter plot with latitude, longitude, and PCA-based color
x, y = map(df['longitude'].values, df['latitude'].values)
sc = map.scatter(x, y, color=colors, marker='o', edgecolor='k', alpha=0.7)

cbar = fig.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
cbar.set_label('PCA Component 1')

plt.title('First PCA Component of Address Embeddings')
plt.savefig("Italy.png")
plt.show()