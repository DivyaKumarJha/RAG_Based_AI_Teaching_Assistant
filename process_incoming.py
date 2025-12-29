# 4. here we will process incoming queries embeddings and find similar chunks of the text (this is text from that jsons file chunks) using cosine similarity.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests



def create_embedding(text_list):
    # this is local instance of bge-m3 model running on port 11434, i got this from github of bge-m3 api for bge-m3
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model" : "deepseek-r1",
        "model" : "llama3.2",
        "prompt" : prompt,
        "stream" : False,
    })
    response = r.json()
    print("Response:", response)
    return response

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]


# Find similairity between question embedding and chunk embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
top_results = 5

# print(similarities)

max_idx = similarities.argsort()[::-1][0:top_results]  # Top 3 most similar chunks

# print(max_idx)

new_df = df.iloc[max_idx]
# print(new_df[['number', 'title', 'text']])


prompt = f'''
I am teaching Descrete Mathematics through videos.
Here are video subtitle chunks containing video titles, video numbers,start time in seconds, end time in seconds and text at that time, and an incoming user query:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}
-----------------------
"{incoming_query}"
user asked this question related to the video chunks, you have to answer in a human way(dont mention the above format it's just for you) where and how much content is taught in which video (in which video and at what time along with timestamps) and guide the user to go to that video chunk to learn more about it from that particular video. If user askes unrelated question, politely tell them that you can only answer questions related to the course.
'''

with open("prompt.txt","w",encoding="utf-8") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print("Final Response:", response)

with open("response.txt","w",encoding="utf-8") as f:
    f.write(response)


# for index, item in new_df.iterrows():
#     print(index, item['text'], item['number'], item['title'], item['start'], item['end'])
