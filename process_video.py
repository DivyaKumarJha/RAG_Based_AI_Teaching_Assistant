# 1. convert videos to mp3 using ffmpeg

import os
import subprocess

files = os.listdir("videos")
# print(files)
for file in files:
    tutorial_number = file.split("lecture-")[1].split("-")[0]
    print(tutorial_number)
    file_name = file.split("lecture-")[0] + "lecture-" + tutorial_number
    print(file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])