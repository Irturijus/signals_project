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
    #                     the beat is groped with all beats in front of it until
    #                     a zero sample (sample without a beat).

    for i in range(beat_array.shape[0]):
        if beat_array[i] > 0:
            if grouping_samples != None:
                j = i+grouping_samples
            else:
                j = i
                while j < beat_array.shape[0] and np.sum(beat_array[j:j+5]) > 0:
                    j += 1
                j += 1
            sum = np.sum(beat_array[i:j])
            beat_array[i:j] = 0
            beat_array[i] = sum

def find_beats_mono(mono_audio: np.ndarray, samplerate: float, beat_samplerate) -> np.ndarray:
    # This function takes an audio signal in the time domain and its sample rate
    # and then finds beats in the song using the total positive frequency spectrum change as an
    # indicator, "marking" them in another array of zeros, effectively creating another 
    # time-domain signal, of which the sample rate is beat_samplerate.
    #
    # Parameters:
    #   mono_audio: the single-channel audio signal
    #   samplerate: the sampling rate in samples/s of the audio signal
    #   beat_samplerate: the desired sampling rate of the beats array

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

    IS_constant = 30
    filter_size = 1 # [s]
    IS_factor = 1.5

    positive_rates_of_change = np.maximum(0, rates_of_change-IS_constant)

    mean_positive_r_o_c = np.mean(positive_rates_of_change, axis=1)

    insensitivity = uniform_filter1d(mean_positive_r_o_c, size=int(beat_samplerate*filter_size))*IS_factor

    beat_array = mean_positive_r_o_c - insensitivity
    beat_array = np.maximum(0, beat_array)

    return beat_array, np.mean(positive_rates_of_change)

def find_beats(audio, samplerate, beat_samplerate):
    # This function takes a stereo audio (song) signal in the time domain and its sample rate
    # and then finds beats in the song, "marking" them in another array of zeros,
    # effectively creating another time-domain signal, of which the sample rate is beat_samplerate.
    # This function calls find_beats_mono for both channels separately and takes the mean of the outputs.
    # It then groups the beats, and removes negligible beats after that.
    #
    # Parameters:
    #   audio: the stereo audio signal
    #   samplerate: the sampling rate in samples/s of the audio signal
    #   beat_samplerate: the desired sampling rate of the beats array

    sufficient_strength_threshold = 0.3

    audio_ch1 = audio[:, 0]
    audio_ch2 = audio[:, 1]

    beats_array_ch1, mean_beat_strength_ch1 = find_beats_mono(audio_ch1, samplerate, beat_samplerate)
    beats_array_ch2, mean_beat_strength_ch2 = find_beats_mono(audio_ch2, samplerate, beat_samplerate)

    beats_array = beats_array_ch1 + beats_array_ch2
    mean_beat_strength = mean_beat_strength_ch1 * 0.5 + mean_beat_strength_ch2 * 0.5

    group_beats(beats_array)

    sufficient_strength_mask = beats_array > sufficient_strength_threshold*mean_beat_strength

    beats_array = beats_array * sufficient_strength_mask

    return beats_array

def display_brightness(brightness, samplerate, decay_rate=5000):
    #visualizes the beats array as a bar in the console that updates in place without
    #scrolling, where the size of the bar represents how much of a beat there is
    #this function will not be in the final version of the project, it is just for testing

    frame_duration = 1.0 / samplerate
    start_time = time.perf_counter()

    FULL_CHAR = '█'
    current_bar_length = 0.0

    for i, target_value in enumerate(brightness):
        target_value_int = max(0, int(target_value*0.1))  # Ensure non-negative

        if current_bar_length > target_value_int:
            decay_factor = 1 - decay_rate * frame_duration
            decay_factor = max(0, decay_factor)  # Prevent negative factor
            current_bar_length *= decay_factor
            current_bar_length = max(target_value_int, current_bar_length)
        else:
            current_bar_length = target_value_int  # Instant rise

        bar = FULL_CHAR * int(round(current_bar_length))
        sys.stdout.write(f"\r{bar}\x1b[K")
        sys.stdout.flush()

        next_frame_time = start_time + (i + 1) * frame_duration
        sleep_time = next_frame_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\nDone.")

def display_brightness_stack(brightness, samplerate, threshold=0.5, max_bars=10, gain = 0.001):
    #visualizes the beats array as a vertically scrolling "screen" of bars, where the
    #size of the bars represent how much of a beat there is
    #this function will not be in the final version of the project, it is just for testing

    frame_duration = 1.0 / samplerate
    start_time = time.perf_counter()

    FULL_CHAR = '='
    MAX_LENGTH = 200  # Max bar width

    bar_stack = []  # Each bar is just a length

    def clear_and_reset_cursor():
        sys.stdout.write("\033[H")  # Cursor to top-left
        sys.stdout.flush()

    def render_bar(length):
        return (MAX_LENGTH//2-length//2) * " " + FULL_CHAR * length

    # Reserve 10 lines in terminal
    sys.stdout.write("\n" * max_bars)
    sys.stdout.flush()

    accumulation = 0

    for i, value in enumerate(brightness):
        # Add new bar if threshold exceeded
        if value > threshold:
            length = (value - threshold) * MAX_LENGTH * gain
        else:
            length = 0
        
        accumulation += length

        interval = int(samplerate/32)

        if i % interval == 0:
            avg_length = accumulation/interval
            accumulation = 0
            bar_stack.insert(0, int(avg_length))
            bar_stack = bar_stack[:max_bars]


        # Redraw all lines
        clear_and_reset_cursor()
        stdout = ""
        for length in bar_stack:
            stdout += render_bar(length).ljust(MAX_LENGTH) + "\n"
        for _ in range(max_bars - len(bar_stack)):
            stdout += " " * MAX_LENGTH + "\n"

        sys.stdout.write(stdout)
        sys.stdout.flush()

        # Frame sync
        next_frame_time = start_time + (i + 1) * frame_duration
        sleep_time = next_frame_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Done.")

def display_brightness_threshold(brightness, samplerate, threshold, decay_rate=40):
    #visualizes the beats array as 10 bars that light up randomly when the beat strength in the beats
    #array exceeds the threshold.
    #this function will not be in the final version of the project, it is just for testing

    frame_duration = 1.0 / samplerate
    start_time = time.perf_counter()

    NUM_SECTIONS = 10
    SECTION_WIDTH = 20
    TOTAL_WIDTH = NUM_SECTIONS * SECTION_WIDTH

    # Unicode characters for different brightness levels (dark to bright)
    LEVELS = [' ', '░', '▒', '▓', '█']
    num_levels = len(LEVELS)

    section_brightness = np.zeros(NUM_SECTIONS, dtype=np.float32)

    index = 0

    for i, value in enumerate(brightness):
        value = float(value)
        now = time.perf_counter()

        # Exponential decay
        decay_factor = max(0.0, 1.0 - decay_rate * frame_duration)
        section_brightness *= decay_factor

        # Trigger: reset lowest section to full brightness if over threshold
        if value > threshold:
            index += 1
            if index > NUM_SECTIONS-1:
                index = 0
            section_brightness[index] = 1.0

        # Build the display string
        display_line = ""
        for b in section_brightness:
            level_idx = int(round(b * (num_levels - 1)))
            level_idx = min(level_idx, num_levels - 1)
            char = LEVELS[level_idx]
            display_line += char * SECTION_WIDTH

        sys.stdout.write(f"\r{display_line}\x1b[K")
        sys.stdout.flush()

        # Real-time sync
        next_frame_time = start_time + (i + 1) * frame_duration
        sleep_time = next_frame_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\nDone.")


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

def strip_array(arr: np.ndarray) -> np.ndarray: #by chatGPT,
    # this function removes the leading and trailing zeros of a numpy array,
    # making it shorter. It returns the stripped array.
    nonzero_indices = np.nonzero(arr)[0]

    if nonzero_indices.size > 0:
        stripped = arr[nonzero_indices[0] : nonzero_indices[-1] + 1]
    else:
        stripped = np.array([])
