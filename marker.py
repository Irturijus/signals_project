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
    # This function takes an audio signal in the time domain and its sample rate
    # and then finds beats in the song, "marking" them in another array of zeros,
    # effectively creating another time-domain signal, of which the sample rate is beat_samplerate

    window_size = int(samplerate*0.05) #the size of the moving part of the audio signal for computing the FFT
    n_fft = int(window_size*1) # the number of samples to take from the DTFT when computing the FFT
    grouping_time = 1/8 #the time difference in seconds below which to group beats into a single beat

    audio_RMS = np.sqrt(np.mean(np.square(mono_audio))) #computes the Root-Mean-square value of the audio signal to determine how loud it is on average
    print(audio_RMS)
    padded_audio = np.pad(mono_audio, (window_size, 0), mode='constant', constant_values=0)/audio_RMS
    #pad the audio with window_size zeros in the beginning, for the fft window to have room to start, and then divide by audio_RMS to normalize it
    total_time = padded_audio.shape[0]/samplerate #the duration of the padded audio signal
    audio_time = mono_audio.shape[0]/samplerate # the duration of the original audio signal
    beats_array = np.zeros(shape=(int(audio_time*beat_samplerate)), dtype=int) #the array in which to store the beat signal in time domain
    weighted_magnitudes = np.zeros(shape=(mono_audio.shape[0], n_fft//2), dtype=np.float32) # the array in which to store the FFT magnitudes from each FFT calulation
    tpc = np.zeros(shape=(mono_audio.shape[0],), dtype=np.float32) # the array in which to store the total positive change in the spectrum with respect to time

    window = np.hanning(window_size) #create a hanning window to smooth out the sharp changes at the beginning and end of the relevant extract of the audio signal
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / samplerate) # get the frequency of each value in the FFT result array
    log_weights = np.log1p(frequencies[1:]) # create an array of logarithmic weights to adjust importance of frequencies for the beat detection, based on which 
    # frequencies the human ear is most senitive to

    for i, sample in enumerate(list(padded_audio)):
        if i < window_size: 
            continue

        frame = padded_audio[i-window_size:i] * window # extact a frame (an extract) from the audio signal
        magnitudes = np.abs(fft(frame, n_fft))[:n_fft // 2] #compute the magnitudes of the positive FFT frequencies and put them into an array

        weighted_magnitudes[i-window_size] = magnitudes * log_weights # apply the logarthmic weights to those magnitudes

        if i % (samplerate//2) == 0: #this if statement is set up to trigger every computed half-second of the signal
            time = (i-window_size)/samplerate # what time of the audio is already processed
            print(f"{(time / audio_time) * 100:.2f}%") # print the progress in %

        if (i - window_size) <= 1: # continue if there is less than two weighted magnitude 1D arrays computed (which means the difference cannot be computed yet)
            continue

        magnitude_difference = weighted_magnitudes[i-window_size]-weighted_magnitudes[i-window_size-1] #compute the difference between the current magnitude 1D array and the previous one
        # (each element in the array is the strength of a specific frequency in the audio signal at a specific time)

        magnitude_difference_relu = np.maximum(0, magnitude_difference) # apply the ReLu function to the difference (make all the negative differences zero)
        tpc[i-window_size] = np.sum(magnitude_difference_relu) # sum all the positive differences from last sample and set the output signal to this value at this time

    del weighted_magnitudes # delete the weigted magnitudes array to free up memory (this array is not needed anymore)

    print(f"{beats_array.shape[0]/beat_samplerate}s") # print the duration of the signal in which the beats were detected

    resampled_tpc = resample_poly(tpc, beat_samplerate, samplerate) # resample the output signal from the input signal's sample rate to the desired beat_samplerate

    # smooth the resampled tpc (total postive change) two times, the first being smoothed more and the other one smoothed less.
    smoothed_tpc = median_filter(resampled_tpc, size=int(samplerate * 5 / (samplerate / beat_samplerate)))
    smoothed_tpc_2 = median_filter(resampled_tpc, size=int(samplerate * 0.05 / (samplerate / beat_samplerate)))

    threshold = 2 # threshold determined by trial and error. The lower this threshold, the smaller changes are registered as beats.
    onsets = np.maximum(0, smoothed_tpc_2-smoothed_tpc-threshold, dtype=float) # the beats array (beats signal in time domain, 0 means no beat, the higher the sample the stronger the beat)

    grouping_samples = int(grouping_time*beat_samplerate) # the length of the grouping window in samples

    # All beats in the grouping window get grouped into one at the beginning of the window. 
    # This window is moved further and the procedure is repeated.

    for i, onset in enumerate(onsets):
        max_index = min(onsets.shape[0]-1, i+grouping_samples) # the index where the grouping window ends
        leading_sum = np.sum(onsets[i:max_index]) # the sum of the leading beat strengths until max_index
        if onset != 0 and leading_sum != 0: # if there is a beat at this moment and there are beats between it and max_index,
            onsets[i] += leading_sum # sum the leading beat strengths and add them to the current beat
            onsets[i+1:max_index] = 0.0 # and set them to zero

    return onsets

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
song_start_time = 0 # [s]
song_end_time = 10 # [s], set to -1 for (almost) full duration

beats_array = find_beats(audio_mono[samplerate*song_start_time:samplerate*song_end_time], samplerate, beat_samplerate) # find the beats

#plt.plot(beats_array)
#plt.show()

beat_data = {
    "beats": list(beats_array),
    "beat_samplerate": beat_samplerate,
    "song_start_time": song_start_time,
    "song_end_time": song_end_time,
    "ogg_file_name": ogg_file_name
} # put the beat data of this song into a dictionary
 
with open("process_song/beats_array.json", "w") as f:
    f.write(json.dumps(beat_data, indent=4)) # save the dictionary as a JSON file
