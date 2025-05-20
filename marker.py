import os
import numpy as np
import matplotlib.pyplot as plt
from lib import *
import json

ogg_file_name = None
for file in os.listdir("process_song"):
    if file.endswith(".ogg"):
        ogg_file_name = file # get the name of the .ogg sound file in the folder "process_song"

if ogg_file_name == None:
    print("Error: No OGG file in process_song")
    exit()

audio, samplerate = load_audio_ffmpeg(f"process_song/{ogg_file_name}") # load the audio file

beat_samplerate = 100 # [samples/s]

beats_array = find_beats(audio, samplerate, beat_samplerate) # find the beats

plt.plot(beats_array)
plt.show()

beat_data = {
    "beats": list(beats_array),
    "beat_samplerate": beat_samplerate,
    "ogg_file_name": ogg_file_name
} # put the beat data of this song into a dictionary
 
with open("process_song/beats_array.json", "w") as f:
    f.write(json.dumps(beat_data, indent=4)) # save the dictionary as a JSON file
