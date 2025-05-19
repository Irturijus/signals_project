import os
from scipy.signal import resample_poly
import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt
from lib import *
from scipy.signal import resample
import numpy as np
from scipy.ndimage import uniform_filter1d
import json

def find_beats(mono_audio: np.ndarray, samplerate: float, beat_samplerate) -> np.ndarray:
    # This function takes an audio signal in the time domain and its sample rate
    # and then finds beats in the song, "marking" them in another array of zeros,
    # effectively creating another time-domain signal, of which the sample rate is beat_samplerate

    total_time = mono_audio.shape[0]/samplerate
    audio_RMS = np.sqrt(np.mean(np.square(mono_audio))) #computes the Root-Mean-square value of the audio signal to determine how loud it is on average

    beat_array_num_samples = int(total_time*beat_samplerate)
    
    beat_array = np.zeros((beat_array_num_samples,), dtype=float)

    audio_sample_size = int(samplerate/beat_samplerate)
    resolution = 16
    fft_sample_size = audio_sample_size * resolution

    frequencies = np.fft.rfftfreq(fft_sample_size, d=1.0/samplerate)[1:]

    prev_rfft_magnitude = np.zeros((fft_sample_size // 2,))

    rates_of_change = np.zeros((beat_array_num_samples, fft_sample_size // 2))
    
    window = np.hamming(audio_sample_size)

    for i in range(beat_array_num_samples-1):
        sample_start = audio_sample_size*i
        sample_end = audio_sample_size*(i+1)
        
        audio_sample = mono_audio[sample_start:sample_end] * window * (1/audio_RMS)

        padded_sample = np.pad(audio_sample, (0, fft_sample_size - audio_sample_size), mode='constant', constant_values=0)

        rfft_magnitude = np.abs(rfft(padded_sample)[1:])
        rfft_magnitude_change = rfft_magnitude - prev_rfft_magnitude
        prev_rfft_magnitude = rfft_magnitude * 1.0

        time_change = 1.0/beat_samplerate

        rates_of_change[i] = rfft_magnitude_change/time_change

    positive_rates_of_change = np.maximum(0, rates_of_change)

    beat_array = np.mean(positive_rates_of_change, axis=1) - np.mean(positive_rates_of_change)*1.6 # dynamically adjust this?
    beat_array = np.maximum(0, beat_array)

    grouping_samples = int(beat_samplerate/8)

    for i in range(beat_array.shape[0]):
        if beat_array[i] > 0:
            sum = np.sum(beat_array[i:i+grouping_samples])
            beat_array[i:i+grouping_samples] = 0
            beat_array[i] = sum

    return beat_array

ogg_file_name = None
for file in os.listdir("process_song"):
    if file.endswith(".ogg"):
        ogg_file_name = file # get the name of the .ogg sound file in the folder "process_song"

if ogg_file_name == None:
    print("Error: No OGG file in process_song")
    exit()

audio, samplerate = load_audio_ffmpeg(f"process_song/{ogg_file_name}") # load the audio file

audio_mono = np.mean(audio, axis=1) # create a mono version of the audio signal for processing

beat_samplerate = 100 # [samples/s]

beats_array = find_beats(audio_mono, samplerate, beat_samplerate) # find the beats

plt.plot(beats_array)
plt.show()

beat_data = {
    "beats": list(beats_array),
    "beat_samplerate": beat_samplerate,
    "ogg_file_name": ogg_file_name
} # put the beat data of this song into a dictionary
 
with open("process_song/beats_array.json", "w") as f:
    f.write(json.dumps(beat_data, indent=4)) # save the dictionary as a JSON file
