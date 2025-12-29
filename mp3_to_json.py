# 2. here in this code, we will transcribe audio files (mp3) into text (json) using Whisper and save them with timestamps into JSON files.
# you can find those text files in jsons folder.

import whisper
import json
import os

# from whisper.whisper import model

audios = os.listdir("audios")

model = whisper.load_model("medium")

for idx, audio in enumerate(audios,2):
    if("_" in audio):
        number = audio.split("_",1)[0]
        title = audio.split("_",1)[1].split(".mp3")[0]
        print(f"\n[{idx}/{len(audios)}] Processing: {number} - {title}")
        result = model.transcribe(audio = f"audios/{audio}", task = "translate", language = "en", word_timestamps= True, verbose=True)
        
        chunks = []

        for segments in result["segments"]:
            chunk = {}
            chunk["number"] = number
            chunk["title"] = title
            chunk["start"] = segments["start"]
            chunk["end"] = segments["end"]
            chunk["text"] = segments["text"].strip()
            chunks.append(chunk)    
        
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
        
        
        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:  # Added encoding
            json.dump(chunks_with_metadata, f, indent=4, ensure_ascii=False)  # Added ensure_ascii
        
        print(f"Saved: jsons/{audio}.json")