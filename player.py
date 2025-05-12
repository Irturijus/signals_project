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
    beat_data= json.load(file)

ogg_file_name = beat_data["ogg_file_name"]
beats_array = beat_data["beats"]
beat_samplerate = beat_data["beat_samplerate"]
start = beat_data["song_start_time"]
end = beat_data["song_end_time"]
 
audio, samplerate = load_audio_ffmpeg(f"process_song/{ogg_file_name}")

sd.play(audio[samplerate*start:samplerate*end]*0.4, samplerate)
display_brightness(beats_array, beat_samplerate, 50)
sd.stop()