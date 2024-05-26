import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from scipy.stats import pearsonr

from scipy.spatial.distance import pdist, squareform
import numpy as np

def lat_lon_to_cartesian(lat, lon):
    """
    Convert latitude and longitude to Cartesian coordinates.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return x, y, z

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

def cosine_distance(vec1, vec2):
    """
    Compute the cosine distance between two vectors.
    Cosine distance is defined as 1 - cosine similarity.
    """
    return 1 - cosine_similarity(*vec1, *vec2)


# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet("embeddings_jina.parquet")

results = {}

whole = df
for country_code in whole["country_code"].unique():
    df = whole[whole["country_code"] == country_code] 
    df.index = pd.RangeIndex(start=0, stop=len(df))
    df['cartesian'] = df.apply(lambda row: lat_lon_to_cartesian(row['latitude'], row['longitude']), axis=1)

    dist_matrix_1 = squareform(pdist(df[['cartesian']], metric=cosine_distance))
    dist_matrix_2 = squareform(pdist(df[['concat_embeddings']], metric=cosine_distance))

    dist_array_1 = dist_matrix_1.flatten()
    dist_array_2 = dist_matrix_2.flatten()

    if len(dist_array_1) < 5:
        continue
    # quotients = [dist_array_2[k] / dist_array_1[k] for k in range(len(dist_array_1)) if dist_array_1[k] != 0]

    # print(f"Lipschitz: {max(quotients)}, Average: {np.average(quotients)}")
    # Calculate the correlation between the flattened distance arrays
    correlation, _ = pearsonr(dist_array_1, dist_array_2)

    print(f"Correlation between the distances: {correlation}")
    results[country_code] = correlation
print(results)
