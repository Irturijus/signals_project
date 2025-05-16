import os
import scipy
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import sounddevice as sd
from lib import *
from scipy.signal import iirnotch, filtfilt, resample
import numpy as np
from scipy.ndimage import median_filter
import json

with open("process_song/beats_array.json", "r") as file:
    beat_data= json.load(file) # load the beat data

ogg_file_name = beat_data["ogg_file_name"]
beats_array = beat_data["beats"]
beat_samplerate = beat_data["beat_samplerate"]
start = beat_data["song_start_time"]
end = beat_data["song_end_time"]
 
audio, samplerate = load_audio_ffmpeg(f"process_song/{ogg_file_name}") # load the song that the beat data belongs to

sd.play(audio[samplerate*start:samplerate*end]*0.4, samplerate) # start playing the song (non-blocking)
display_brightness_threshold(beats_array, beat_samplerate, 10, 40) # start visualizing the beats signal (blocking)
sd.stop() # stop playing the song after the data has been visualized