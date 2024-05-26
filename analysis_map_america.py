import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import colorsys

def hsv_to_rgb(x, _, y):
    """
    Convert HSV with hue x, value y, and saturation 1 to an RGB value.
    
    Parameters:
    x (float): Hue value, between 0 and 1.
    y (float): Value value, between 0 and 1.
    
    Returns:
    tuple: Corresponding RGB value with components scaled between 0 and 255.
    """
    # Saturation is fixed at 1
    saturation = 1.0
    hue = x
    value = y
    
    # Convert HSV to RGB using colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Scale RGB values to 0-255 range
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    
    return (r, g, b)

# Read the Parquet file into a pandas DataFrame
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.decomposition import PCA 
from mpl_toolkits.basemap import Basemap 

df = pd.read_parquet("embeddings_jina.parquet") 
# country_counts = df['country'].value_counts()
# print(country_counts)
df = df[df["country_code"] == "KR"] 
embeddings = np.array(df['concat_embeddings'].tolist()) 
# Perform PCA and keep two components 
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings) 

pca_component1 = reduced_embeddings[:, 0] 
pca_component2 = reduced_embeddings[:, 1]
#pca_component3 = reduced_embeddings[:, 2]

fig, ax = plt.subplots(figsize=(15, 10))
margin = 5
map = Basemap(ax=ax, llcrnrlon=min(df['longitude']) - margin,llcrnrlat=min(df['latitude']) - margin,urcrnrlon=max(df['longitude']) + margin,urcrnrlat=max(df['latitude']) + margin)#, llcrnrlon=-131,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51) 
map.drawcoastlines() 
map.drawcountries() # Use a 2D colormap. We'll combin the values for both components 
norm1 = plt.Normalize(vmin=pca_component1.min(), vmax=pca_component1.max()) 
norm2 = plt.Normalize(vmin=pca_component2.min(), vmax=pca_component2.max())
#norm3 = plt.Normalize(vmin=pca_component3.min(), vmax=pca_component3.max()) 
# We will use a simple approach to generate a 2D colormap: 
# # combine normalized PCA components into a single color array. 
colors = np.vstack((norm1(pca_component1), np.zeros(pca_component1.shape), norm2(pca_component2))).T
colors = [tuple(x) for x in colors]
# colors = plt.cm.jet(colors) # Convert to RGB using jet colormap 
x, y = map(df['longitude'].values, df['latitude'].values) 
sc = map.scatter(x, y, c=colors, marker='.', s=1000, alpha=0.75) 
plt.title('First 2 PCA Components of Address Embeddings')
# Creating a color bar for a bivariate color map is complex and may not be straightforwardly informative. 
# # Hence we skip creating a color bar for this bivariate map, because it's not as simple 
# as mapping a range of values to a single color dimension. 
plt.show()