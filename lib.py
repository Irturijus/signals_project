import sys
import time
import subprocess
import numpy as np
import random

def display_brightness(brightness, samplerate, decay_rate=5000):
    frame_duration = 1.0 / samplerate
    start_time = time.perf_counter()

    FULL_CHAR = '█'
    current_bar_length = 0.0

    for i, target_value in enumerate(brightness):
        target_value_int = max(0, int(target_value*10))  # Ensure non-negative

        target_value_int

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

def display_brightness_stack(
    brightness, samplerate, threshold=0.5,
    max_bars=10
):
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
            length = (value - threshold) * MAX_LENGTH * 0.01
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


def get_audio_samplerate(file_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return int(result.stdout.decode().strip())

def load_audio_ffmpeg(file_path):
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
