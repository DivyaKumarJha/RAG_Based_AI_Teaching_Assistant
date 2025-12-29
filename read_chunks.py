# 3. text to vector, text ki semantic meaning ko rakhte hue unhe vector mein convert karna
# cosine similarity is 0 when vectors are perpendicular matlab statements opposite hai, and 1 when they are identical, malab statements same hai 

# so embeddings is way to represnt a statement is high dimensional vector.
# SO HERE WE WILL CONVERT THOSE TEXT CHUNKS INTO EMBEDDINGS USING BGE-M3 MODEL AND SAVE THEM INTO A DATAFRAME USING JOBLIB

import http
import requests
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity



def create_embedding(text_list):
    # this is local instance of bge-m3 model running on port 11434, i got this from github of bge-m3 api for bge-m3
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
    })
    embedding = r.json()["embeddings"]
    return embedding



jsons = os.listdir("new_merged_jsons") # List all the files in the jsons directory


my_dict = []
chunk_id = 0


for json_file in jsons:
    with open(f"new_merged_jsons/{json_file}") as f:
        content = json.load(f)
        
    print("creating embeddings for " + json_file)    
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    
    for i, chunk in enumerate(content['chunks']):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id+=1
        my_dict.append(chunk)
        

df = pd.DataFrame.from_records(my_dict)
# save this dataframe using joblib
joblib.dump(df, "embeddings.joblib")

