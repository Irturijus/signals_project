import os
import json
import random
import shutil
import numpy as np

with open('process_song/beats_array.json', 'r') as file:
    data = json.load(file)

beats = data['beats']
samplerate = data["beat_samplerate"]
songFilename = data['ogg_file_name']
bpm = data['tempo']
name = 'Unknown Song'
author = 'Unknown Author'
output_folder = 'beatmap'


duration = len(beats) / samplerate
total_beats = sum(1 for i in beats if i > 0)
print(
    f'Samplerate: {samplerate},\n Duration: {duration}, \n Total beats: {total_beats}, \n BPM: {bpm}')

nps = total_beats/duration

easy_nps = 0.8
normal_nps = 1.8
hard_nps = 3
expert_nps = 4.5
expertplus_nps = 6

easy_generator = False
normal_generator = False
hard_generator = False
expert_generator = False
expertplus_generator = False

if nps < expertplus_nps:
    expertplus_generator = True
    expertplus_exists = False
else:
    expertplus_exists = True
    
if nps < expert_nps:
    expert_generator = True
    expert_exists = False
else:
    expert_exists = True

if nps < hard_nps:
    hard_generator = True
    hard_exists = False
else:
    hard_exists = True

if nps < normal_nps:
    normal_generator = True
    normal_exists = False
else:
    normal_exists = True

if nps < easy_nps:
    easy_generator = True
    easy_exists = False
    print("Map cannot be created")
    exit
else:
    easy_exists = True

multiplier = 0
tolerance = 0.2

while multiplier < 0.3 and not (expertplus_generator and expert_generator and hard_generator and normal_generator and easy_generator):

    beats = np.maximum(0, beats-multiplier*np.max(beats))

    total_beats = sum(1 for i in beats if i > 0)

    nps = total_beats/duration

    if nps > (easy_nps - tolerance) and nps < (easy_nps + tolerance) and not easy_generator:
        easy = beats
        easy_generator = True
        print("easy difficulty generated")
    
    elif nps > (normal_nps - tolerance) and nps < (normal_nps + tolerance) and not normal_generator:
        normal = beats
        normal_generator = True
        print("normal difficulty generated")

    elif nps > (hard_nps - tolerance) and nps < (hard_nps + tolerance) and not hard_generator:
        hard = beats
        hard_generator = True
        print("hard difficulty generated")

    elif nps > (expert_nps - tolerance) and nps < (expert_nps + tolerance) and not expert_generator:
        expert = beats
        expert_generator = True
        print("expert difficulty generated")

    elif nps > (expertplus_nps - tolerance) and nps < (expertplus_nps + tolerance) and not expertplus_generator:
        expertplus = beats
        expertplus_generator = True
        print("expert+ difficulty generated")

    multiplier += 0.00001

def seconds_to_beat(seconds):
    return (seconds / 60) * bpm


def generate_info():
    info = {
        'version': '4.0.0',
        'song': {
            'title': name,
            'author': author
        },
        'audio': {
            'bpm': bpm,
            'songFilename': songFilename,
        },
        'coverImageFilename': 'cover.png',
        '_environmentName': 'DefaultEnvironment',
        '_allDirectionsEnvironmentName': 'GlassDesertEnvironment',

        'difficultyBeatmaps': [
            {
                'difficulty': 'Easy',
                'difficultyRank': 1,
                'beatmapDataFilename': 'Easy.dat',
                'noteJumpMovementSpeed': 10,
                'noteJumpStartBeatOffset': 0,
                'colorSchemeId': 'Default'
            },
            {
                'difficulty': 'Normal',
                'difficultyRank': 3,
                'beatmapDataFilename': 'Normal.dat',
                'noteJumpMovementSpeed': 10,
                'noteJumpStartBeatOffset': 0,
                'colorSchemeId': 'Default'
            },
            {
                'difficulty': 'Hard',
                'difficultyRank': 5,
                'beatmapDataFilename': 'Hard.dat',
                'noteJumpMovementSpeed': 10,
                'noteJumpStartBeatOffset': 0,
                'colorSchemeId': 'Default'
            },
            {
                'difficulty': 'Expert',
                'difficultyRank': 7,
                'beatmapDataFilename': 'Expert.dat',
                'noteJumpMovementSpeed': 10,
                'noteJumpStartBeatOffset': 0,
                'colorSchemeId': 'Default'
            },
            {
                'difficulty': 'Expert+',
                'difficultyRank': 9,
                'beatmapDataFilename': 'Expertplus.dat',
                'noteJumpMovementSpeed': 10,
                'noteJumpStartBeatOffset': 0,
                'colorSchemeId': 'Default'
            },
        ],
    }

    return info


def generate_easy(beats):
    color_notes = []

    max_val = max(beats)

    possible_blocks = {(0, 2, 4), (1, 2, 0), (2, 2, 0), (3, 2, 5),
                       (0, 1, 2),                       (3, 1, 3),
                       (0, 0, 6), (1, 0, 1), (2, 0, 1), (3, 0, 7)}

    banned_blocks = set()

    for i, val in enumerate(beats):
        if not val:
            continue

        cur_blocks = []

        second = i / samplerate
        beat = seconds_to_beat(second)

        if val > 0.75 * max_val:
            note_count = 3
        elif val > 0.6 * max_val:
            note_count = 2
        else:
            note_count = 1

        available_blocks = possible_blocks - banned_blocks
        cur_block = random.choice(list(available_blocks))
        cur_blocks.append(cur_block)

        if cur_block[0] in [0, 1]:
            color = 0
        else:
            color = 1

        dot_roll = random.randint(0, 7)  # Make some blocks any direction
        direction = 8 if dot_roll == 7 else cur_block[2]

        if note_count == 1:
            color_notes.append({
                'b': round(beat, 3),  # Beat
                'x': cur_block[0],  # Line Index
                'y': cur_block[1],  # Line Layer
                'a': 0,
                'c': color,
                'd': direction,  # Cut direction
            })
            cur_blocks.append((cur_block[0], cur_block[1], direction))

        # Wall builder
        banned_directions = {x[2] for x in banned_blocks}
        possible_wall_directions = [d for d in [
            0, 1] if d not in banned_directions]
        banned_blocks = set()

        if possible_wall_directions:
            wall_direction = random.choice(possible_wall_directions)

            if note_count > 1:
                for i in range(0, note_count):
                    color_notes.append({
                        'b': round(beat, 3),  # Beat
                        'x': cur_block[0],  # Line Index
                        'y': 0 + i,  # Line Layer
                        'a': 0,
                        'c': color,
                        'd': wall_direction,  # Cut direction
                    })
                    cur_blocks.append((cur_block[0], 0+i, wall_direction))
                    banned_blocks.add((cur_block[0], 0, wall_direction))

        # Ban blocks adjacent to this one
        for cur in cur_blocks:
            for block in possible_blocks:
                if (abs(block[0] - cur[0]) + abs(block[1] - cur[1])) < 2:
                    banned_blocks.add(block)

    return {
        'version': '3.3.0',
        'colorNotes': color_notes,
        'bombNotes': [],
        'obstacles': [],
        'sliders': [],
        'burstSliders': [],
        'waypoints': [],
        'basicBeatmapEvents': [],
        'colorBoostBeatmapEvents': [],
        'lightColorEventBoxGroups': [],
        'lightRotationEventBoxGroups': [],
        'lightTranslationEventBoxGroups': [],
        'useNormalEventsAsCompatibleEvents': True
    }


def export_map(a, b, c, d, e):
    os.makedirs(output_folder, exist_ok=True)
    src = 'process_song/' + songFilename
    dst = 'beatmap/' + songFilename
    shutil.copy(src, dst)

    # Info.dat
    info_data = generate_info()
    with open(os.path.join(output_folder, 'Info.dat'), 'w') as f:
        json.dump(info_data, f, indent=2)

    # Easy.dat

    if a:

        easy_data = generate_easy(easy)
        with open(os.path.join(output_folder, 'Easy.dat'), 'w') as f:
            json.dump(easy_data, f, indent=2)

        easy_data == 0

    if b:

        normal_data = generate_easy(normal)
        with open(os.path.join(output_folder, 'Normal.dat'), 'w') as f:
            json.dump(normal_data, f, indent=2)

        normal_data == 0

    if c:

        hard_data = generate_easy(hard)
        with open(os.path.join(output_folder, 'Hard.dat'), 'w') as f:
            json.dump(hard_data , f, indent=2)

        hard_data == 0

    if d:

        expert_data = generate_easy(expert)
        with open(os.path.join(output_folder, 'Expert.dat'), 'w') as f:
            json.dump(expert_data, f, indent=2)

        expert_data == 0

    if e:

        expertp_data = generate_easy(expertplus)
        with open(os.path.join(output_folder, 'Expertplus.dat'), 'w') as f:
            json.dump(expertp_data, f, indent=2)

        expertp_data == 0

    print(f'Exported!')


export_map(easy_exists, normal_exists, hard_exists, expert_exists, expertplus_exists)

shutil.make_archive('beatmap', 'zip', 'beatmap')
