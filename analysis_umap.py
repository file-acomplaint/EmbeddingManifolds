import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt

df = pd.read_parquet("embeddings_unzipped.parquet")
df = df[df['country'].isin(['United States', 'Canada', 'Mexico'])]
# Ensure the embeddings are in the correct format (list of lists)
embeddings = np.vstack(df['concat_embeddings'])

# Apply UMAP
umap_model = UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# Combine the reduced embeddings with the country information
reduced_df = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
reduced_df.index = df.index 
reduced_df['country'] = df['country']  # Now the index alignment should work
reduced_df['latitude'] = df['latitude']  # Add longitude for color scale

# Plot
fig, ax = plt.subplots()

# Define marker shapes for each country
markers = ['o', 's']  # Marker shapes for "United States" and "Canada"
marker_dict = {'United States': 'o', 'Mexico': 's'}  # Dictionary mapping countries to marker shapes

# Iterate over groups and plot
groups = reduced_df.groupby('country')
for name, group in groups:
    ax.scatter(group.x, group.y, marker=marker_dict[name], c=group['latitude'], cmap='viridis', label=name)

# Add colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
cbar.set_label('latitude')

ax.legend()
plt.show()