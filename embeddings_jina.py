import requests
import json

url = 'https://api.jina.ai/v1/embeddings'

headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer jina_14f63b5b2233435496e9917eef414cc6F7gaX7gbT0csfIdDNrznjEnQZ2c9'
}

data = {
  'input': ["Your text string goes here", "You can send multiple texts"],
  'model': 'jina-embeddings-v2-base-en',
  'encoding_type': 'float'
}

import pandas as pd
import numpy as np
import pycountry

# Function to get country name from country code using pycountry
def get_country_name(country_code):
    try:
        country_name = pycountry.countries.get(alpha_2=country_code).name
    except AttributeError:
        country_name = "Unknown"
    return country_name

def get_embeddings(texts, **kwargs):
    #return np.random.random((10))
    # replace newlines, which can negatively affect performance.
    texts = [text.replace("\n", " ") for text in texts]
    texts = [text[0:8190] if len(text) > 8191 else text for text in texts]

    MAX_BATCH_SIZE = 2048
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    # Initialize an empty list to hold all embeddings
    embeddings_data = []
    
    # Process each chunk
    for chunk in chunker(texts, MAX_BATCH_SIZE):
        data["input"] = chunk
        response = requests.post(url, headers=headers, json=data)
        chunk_embeddings = [x["embedding"] for x in json.loads(response.content)["data"]]
        embeddings_data.extend(chunk_embeddings)
    
    return embeddings_data

def append_embeddings_row(df):
    # Concatenate "admin_name1" and "admin_name2" for the entire DataFrame
    concatenated_names = df["admin_name2"] + ", " + df["admin_name1"] + ", " + df["country"]
    # concatenated_names_zip = df["postal_code"] + " " + df["admin_name2"] + ", " + df["admin_name1"] + ", " + df["country"]
    # df['concatenated_zip'] = concatenated_names_zip
    print([df["admin_name1"][k] for k, x in enumerate(concatenated_names) if type(x) != str])
    # Generate embeddings for each concatenated name and store them in a list
    embeddings_list = get_embeddings(list(concatenated_names))
    
    # embedding_strings = [json.dumps(x) for x in embeddings_list]
    df['concat_embeddings'] = embeddings_list
    df['concatenated'] = concatenated_names
    
    return df


# Define the file path
file_path = "allCountries.txt"

# Define column names based on the readme specification
column_names = [
    "country_code", "postal_code", "place_name", "admin_name1",
    "admin_code1", "admin_name2", "admin_code2", "admin_name3",
    "admin_code3", "latitude", "longitude", "accuracy"
]

# Read the tab-delimited file into a pandas DataFrame
df = pd.read_csv(file_path, sep="\t", header=None, names=column_names, encoding="utf-8")

unique_df = df.drop_duplicates(subset= "postal_code", keep="first")
unique_df = unique_df.drop_duplicates(subset= "admin_name2", keep="first")


filtered_df = unique_df.dropna(subset=["admin_name2"])
filtered_df = filtered_df.dropna(subset=["admin_name1"])
test_df = filtered_df
test_df['country'] = test_df['country_code'].apply(get_country_name)
test_df = test_df.dropna(subset=["country"])

test_df.drop(columns=["admin_code1", "admin_code2", "admin_code3"], inplace=True)
test_df = test_df[pd.to_numeric(test_df['postal_code'], errors='coerce').notnull()]
test_df['postal_code'] = test_df['postal_code'].astype(str)
append_embeddings_row(test_df)

test_df.to_parquet('embeddings.parquet', compression='gzip')
