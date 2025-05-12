import os
from scipy.signal import resample_poly
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from lib import *
from scipy.signal import resample
import numpy as np
from scipy.ndimage import median_filter
import json

def find_beats(mono_audio: np.ndarray, samplerate: float, beat_samplerate) -> np.ndarray:
    window_size = int(samplerate*0.05)
    n_fft = int(window_size*0.25)
    grouping_time = 1/8

    audio_RMS = np.sqrt(np.mean(np.square(mono_audio)))
    print(audio_RMS)
    padded_audio = np.pad(mono_audio, (window_size, 0), mode='constant', constant_values=0)/audio_RMS #also volume-normalized
    total_time = padded_audio.shape[0]/samplerate
    audio_time = mono_audio.shape[0]/samplerate
    beats_array = np.zeros(shape=(int(audio_time*beat_samplerate)), dtype=int)
    weighted_magnitudes = np.zeros(shape=(mono_audio.shape[0], n_fft//2), dtype=np.float32)
    spectral_flux = np.zeros(shape=(mono_audio.shape[0],), dtype=np.float32) # total positive change

    window = np.hanning(window_size)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / samplerate)
    log_weights = np.log1p(frequencies[1:]) # log(1 + f)

    for i, sample in enumerate(list(padded_audio)):
        if i < window_size: 
            continue

        frame = padded_audio[i-window_size:i] * window
        magnitudes = np.abs(fft(frame, n_fft))[:n_fft // 2]

        weighted_magnitudes[i-window_size] = magnitudes * 1

        if i % (samplerate//2) == 0:
            time = (i-window_size)/samplerate
            print(f"{(time / audio_time) * 100:.2f}%")

        if (i - window_size) <= 1: 
            continue

        flux = np.maximum(0, weighted_magnitudes[i-window_size]-weighted_magnitudes[i-window_size-1])
        spectral_flux[i-window_size] = np.sum(flux)

    del weighted_magnitudes

    print(f"{beats_array.shape[0]/beat_samplerate}s")

    up = beat_samplerate
    down = samplerate

    resampled_flux = resample_poly(spectral_flux, up, down)

    smoothed_flux = median_filter(resampled_flux, size=int(samplerate * 1 / (samplerate / beat_samplerate)))
    smoothed_flux_2 = median_filter(resampled_flux, size=int(samplerate * 0.02 / (samplerate / beat_samplerate)))

    threshold = 0.2
    onsets = np.maximum(0, smoothed_flux_2-smoothed_flux-threshold, dtype=float)

    grouping_samples = int(grouping_time*beat_samplerate)

    for i, onset in enumerate(onsets):
        max_index = min(onsets.shape[0]-1, i+grouping_samples)
        leading_sum = np.sum(onsets[i:max_index])
        if onset != 0 and leading_sum != 0:
            onsets[i] += leading_sum
            onsets[i+1:max_index] = 0.0

    return onsets

ogg_file_name = None
for file in os.listdir("process_song"):
    if file.endswith(".ogg"):
        ogg_file_name = file

if ogg_file_name == None:
    print("Error: No OGG file in process_song")
    exit()

audio, samplerate = load_audio_ffmpeg(f"process_song/{ogg_file_name}")

audio_mono = np.mean(audio, axis=1)

beat_samplerate = 100
song_start_time = 0
song_end_time = 10

beats_array = find_beats(audio_mono[samplerate*song_start_time:samplerate*song_end_time], samplerate, beat_samplerate)

#plt.plot(beats_array)
#plt.show()

beat_data = {
    "beats": list(beats_array),
    "beat_samplerate": beat_samplerate,
    "song_start_time": song_start_time,
    "song_end_time": song_end_time,
    "ogg_file_name": ogg_file_name
}

with open("process_song/beats_array2.json", "w") as f:
    f.write(json.dumps(beat_data, indent=4))
