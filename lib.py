import sys
import time
import subprocess
import numpy as np
import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def group_beats(beat_array, grouping_samples=None):
    # groups all instances of adjacent beats in beats_array together into single beats.
    # beat_array is modified.
    #
    # Parameters:
    #   beat_array: the array where the beats are stored
    #
    #   grouping_samples: if provided, every beat is grouped with anything
    #                     grouping_samples samples in front of it. Othewise,
    #                     the beat is grouped with all beats in front of it until 6
    #                     consecutive zero samples (samples without a beat).

    for i in range(beat_array.shape[0]):
        if beat_array[i] > 0:
            if grouping_samples != None:
                j = i+grouping_samples
            else:
                j = i
                while j < beat_array.shape[0] and np.sum(beat_array[j:j+6]) > 0:
                    j += 1
                j += 1
            sum = np.sum(beat_array[i:j])
            beat_array[i:j] = 0
            beat_array[i] = sum

def get_spectrum_time_derivatives(mono_audio: np.ndarray, samplerate: float, output_samplerate, resolution=16) -> np.ndarray:
    # computes the vectors (for every discrete time step) of time derivatives of the frequency content in mono_audio, where each entry
    # in a vector represents the time derivative of a specific frequency range at that time in the audio signal. The vectors are sampled
    # at output_samplerate.
    #
    # Parameters:
    #   mono_audio: the single-channel audio signal
    #   samplerate: the sampling rate in samples/s of the audio signal
    #   output_samplerate: the desired sampling rate of the derivative vectors
    #   resolution: the factor by which the zero-padding increases the length of the audio slice
    #   (to extend the frequency range)

    total_time = mono_audio.shape[0]/samplerate
    audio_RMS = np.sqrt(np.mean(np.square(mono_audio))) #computes the Root-Mean-square value of the audio signal to determine how loud it is on average

    out_array_num_samples = int(total_time*output_samplerate)
    
    beat_array = np.zeros((out_array_num_samples,), dtype=float)

    audio_slice_size = int(samplerate/output_samplerate)
    fft_output_size = audio_slice_size * resolution

    frequencies = np.fft.rfftfreq(fft_output_size, d=1.0/samplerate)[1:]

    prev_rfft_magnitude = np.zeros((fft_output_size // 2,))

    rates_of_change = np.zeros((out_array_num_samples, fft_output_size // 2))
    
    window = np.hamming(audio_slice_size)

    for i in range(out_array_num_samples-1):
        slice_start = audio_slice_size*i
        slice_end = audio_slice_size*(i+1)
        
        audio_slice = mono_audio[slice_start:slice_end] * window * (1/audio_RMS)

        padded_slice = np.pad(audio_slice, (0, fft_output_size - audio_slice_size), mode='constant', constant_values=0)

        rfft_magnitude = np.abs(rfft(padded_slice)[1:])
        rfft_magnitude_change = rfft_magnitude - prev_rfft_magnitude
        prev_rfft_magnitude = rfft_magnitude * 1.0

        time_change = 1.0/output_samplerate

        rates_of_change[i] = rfft_magnitude_change/time_change

    return rates_of_change

def derivatives_to_beats(derivatives, filter_size, IS_constant=30, IS_factor=1.5):
    # Processes the given audio spectrum time derivatrives into a beats array (discrete-time signal),
    # of which the non-zero samples represent beats, and their value represents the strength of that beat.
    # The higher the IS_constant and IS_factor, the less sensitive the beat detection becomes.
    #
    # Parameters:
    #   derivatives: the time-domain signal of the spectrum time derivative vectors
    #   filter_size: the size of the mean filter in samples
    #   IS_constant: the constant which is first subtracted from all derivatives
    #   IS_factor: the factor by which the adaptive threshold is multiplied.

    positive_derivatives = np.maximum(0, derivatives-IS_constant)

    mean_positive_r_o_c = np.mean(positive_derivatives, axis=1)

    insensitivity = uniform_filter1d(mean_positive_r_o_c, size=filter_size)*IS_factor

    beats_array = mean_positive_r_o_c - insensitivity
    beats_array = np.maximum(0, beats_array)

    return beats_array

def compute_mean_beat_strength(beats_array):
    # Computes the mean beat strength of the given array containing beats (beats_array).
    num_beats = np.count_nonzero(beats_array)
    beat_sum = np.sum(beats_array)
    mean_beat_strength = num_beats/beat_sum
    return mean_beat_strength

def find_tempo(derivatives, beat_samplerate, tol=3):
    # finds the tempo of a song based on its frequency spectrum time derivatives, using
    # the Discrete Fourier Transform.
    #
    # Parameters:
    #     derivatives: the time-domain signal of the frequency spectrum time derivative vectors
    #     beat_samplerate: the sampling rate of the derivatives, in samples/s
    #     tol: the bilateral tolerance used when finding the presence of tempo for different time signatures
    derivatives_size = derivatives.shape[0]
    positive_derivatives = np.maximum(0, derivatives)
    positive_derivatives_mean =  np.mean(positive_derivatives, axis=1)

    magnitudes = np.abs(rfft(positive_derivatives_mean)[1:])
    frequencies = np.fft.rfftfreq(derivatives_size, d=1.0/beat_samplerate)[1:]
    tempo_values = 60.0*frequencies

    tempo_mask_1 = (tempo_values > 30) # ignore any tempo below 30 BPM
    magnitudes *= tempo_mask_1
    beat_tempo_location = np.argmax(magnitudes)
    ts2_tempo_location = beat_tempo_location//2 #tempo locations for different time signatures
    ts3_tempo_location = beat_tempo_location//3
    ts4_tempo_location = beat_tempo_location//4

    ts_tempo_values = np.zeros((3,), dtype=float)

    #determine the time signature
    ts_tempo_values[0] = np.max(magnitudes[ts2_tempo_location-tol:ts2_tempo_location+tol])
    ts_tempo_values[1] = np.max(magnitudes[ts3_tempo_location-tol:ts3_tempo_location+tol])
    ts_tempo_values[2] = np.max(magnitudes[ts4_tempo_location-tol:ts4_tempo_location+tol])

    tempo_identifier = np.argmax(ts_tempo_values)

    #set the tempo to the right one for the chosen time signature
    if tempo_identifier == 0:
        tempo = tempo_values[ts2_tempo_location]
    elif tempo_identifier == 1:
        tempo = tempo_values[ts3_tempo_location]
    else:
        tempo = tempo_values[ts4_tempo_location]

    return tempo

def find_beats_and_tempo(audio, samplerate, beat_samplerate, sufficient_strength_threshold=0.1, filter_duration=1):
    # This function takes a stereo audio (song) signal in the time domain and its sample rate
    # and then finds beats in the song, "marking" them in another array of zeros,
    # effectively creating another time-domain signal, of which the sample rate is beat_samplerate.
    # This function calls find_beats_mono for both channels separately and takes the mean of the outputs.
    # It then groups the beats, and removes negligible beats after that.
    #
    # Parameters:
    #   audio: the stereo audio signal in time domain
    #   samplerate: the sampling rate in samples/s of the audio signal
    #   beat_samplerate: the desired sampling rate of the beats array
    #   sufficient_strength_threshold: the strength threshold below which to ignore beats.
    #   filter_duration: the size in seconds of the mean filter used for creating the adaptive threshold.

    audio_ch1 = audio[:, 0]
    audio_ch2 = audio[:, 1]

    derivatives_ch1 = get_spectrum_time_derivatives(audio_ch1, samplerate, beat_samplerate)
    derivatives_ch2 = get_spectrum_time_derivatives(audio_ch2, samplerate, beat_samplerate)

    beats_array_ch1 = derivatives_to_beats(derivatives_ch1, int(beat_samplerate*filter_duration))
    beats_array_ch2 = derivatives_to_beats(derivatives_ch2, int(beat_samplerate*filter_duration))
    beats_array = beats_array_ch1 + beats_array_ch2

    group_beats(beats_array)

    mean_beat_strength = compute_mean_beat_strength(beats_array)
    sufficient_strength_mask = beats_array > sufficient_strength_threshold*mean_beat_strength
    beats_array = beats_array * sufficient_strength_mask

    tempo_ch1 = find_tempo(derivatives_ch1, beat_samplerate)
    tempo_ch2 = find_tempo(derivatives_ch1, beat_samplerate)

    mean_tempo = 0.5*tempo_ch1+0.5*tempo_ch2

    return beats_array, mean_tempo

def get_audio_samplerate(file_path): #by chatGPT
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return int(result.stdout.decode().strip())

def load_audio_ffmpeg(file_path): #by chatGPT
    sr = get_audio_samplerate(file_path)

    cmd = [
        "ffmpeg", "-i", file_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "2",  # stereo
        "-loglevel", "error",
        "pipe:1"
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(out.stdout, dtype=np.float32)
    audio = audio.reshape(-1, 2)
    return audio, sr
