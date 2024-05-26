import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet("embeddings_unzipped.parquet")

print(df.columns)
embeddings = np.array(df['concat_embeddings'].tolist())

# Performing PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Creating a scatter plot
plt.figure(figsize=(10, 7))
for country in df['country'].unique():
    if country not in ["United States", "Germany", "France"]:
        continue
    # Boolean index for selecting rows for each country
    indices = df['country'] == country
    plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=country)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of Embeddings, Colored by Country')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.decomposition import PCA

# Assumption: PCA is already performed and `reduced_embeddings` is available
# Perform PCA if not done yet or extract first component
if 'reduced_embeddings' not in locals():
    embeddings = np.array(df['concat_embeddings'].tolist())
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
# Extracting the first PCA component for coloring
pca_component1 = reduced_embeddings[:, 0]

# Plotting
plt.figure(figsize=(15, 10))
fig, ax = plt.subplots(figsize=(15, 10))
map = Basemap(ax=ax)

map.drawcoastlines()
map.drawcountries()

norm = plt.Normalize(vmin=pca_component1.min(), vmax=pca_component1.max())
colors = plt.cm.jet(norm(pca_component1))  # Normalized colors

# Scatter plot with latitude, longitude, and PCA-based color
x, y = map(df['longitude'].values, df['latitude'].values)
sc = map.scatter(x, y, color=colors, marker='o', edgecolor='k', alpha=0.7)

cbar = fig.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
cbar.set_label('PCA Component 1')

plt.title('World Map Color-coded by First PCA Component')
plt.show()