# How to use this RAG AI teaching assistant in your own data

## Step 1 : Collect your videos
Move all your video file to videos folder

## Step 2 : Convert to mp3
Convert all the the video files to mp3 by running video_to_mp3.py file

## Step 3 : Convert mp3 to json
Convert all the mp3 files to json by running mp3_to_json.py

## Step 4 : Convert Json files to Vectors
Use the file read_chunks.py to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 : Prompt Generation and feeding to LLM
Read the joblib file and load it into the memory. Then create a relevant prompt as per the and feed it to the LLM 