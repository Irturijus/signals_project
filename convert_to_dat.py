import os
import json
import random
import shutil

if os.path.exists('beatmap'):
    shutil.rmtree('beatmap')

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


def generate_easy(difficulty):
    color_notes = []

    max_val = max(beats)

    possible_blocks = {(0, 0, 6), (1, 0, 1), (2, 0, 1), (3, 0, 7),
                       (0, 1, 2),                       (3, 1, 3),
                       (0, 2, 4), (1, 2, 0), (2, 2, 0), (3, 2, 5)}

    banned_blocks = set()

    for i, val in enumerate(beats):
        if not val:
            continue

        second = i / samplerate
        beat = seconds_to_beat(second)

        if val > (0.75-(difficulty-1)*0.01) * max_val:
            note_count = 3
        elif val > (0.5-(difficulty-1)*0.01) * max_val:
            note_count = 2
        else:
            note_count = 1

        available_blocks = possible_blocks - banned_blocks

        cur_block = random.choice(list(available_blocks))

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

        # Wall builder
        wall_direction = random.choice([0, 1])
        wall_start = random.choice([0, 2])

        for i in range(0, note_count - 1):
            if wall_start == 0:
                color_notes.append({
                    'b': round(beat, 3),  # Beat
                    'x': cur_block[0],  # Line Index
                    'y': 0 + i,  # Line Layer
                    'a': 0,
                    'c': color,
                    'd': wall_direction,  # Cut direction
                })
            if wall_start == 2:
                color_notes.append({
                    'b': round(beat, 3),  # Beat
                    'x': cur_block[0],  # Line Index
                    'y': 2 - i,  # Line Layer
                    'a': 0,
                    'c': color,
                    'd': wall_direction,  # Cut direction
                })

        # Ban blocks adjacent to this one
        banned_blocks = set()

        for block in possible_blocks:
            if abs(block[0] - cur_block[0]) + abs(block[1] - cur_block[1]) <= 1:
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


def export_map():
    os.makedirs(output_folder, exist_ok=True)
    src = 'process_song/' + songFilename
    dst = 'beatmap/' + songFilename
    shutil.copy(src, dst)

    # Info.dat
    info_data = generate_info()
    with open(os.path.join(output_folder, 'Info.dat'), 'w') as f:
        json.dump(info_data, f, indent=2)

    # Easy.dat
    easy_data = generate_easy(1)
    with open(os.path.join(output_folder, 'Easy.dat'), 'w') as f:
        json.dump(easy_data, f, indent=2)

    easy_data == 0

    # normal_data = generate_easy(3)
    # with open(os.path.join(output_folder, 'Normal.dat'), 'w') as f:
    #     json.dump(normal_data, f, indent=2)

    # normal_data == 0

    # hard_data = generate_easy(5)
    # with open(os.path.join(output_folder, 'Hard.dat'), 'w') as f:
    #     json.dump(hard_data , f, indent=2)

    # hard_data == 0

    # expert_data = generate_easy(7)
    # with open(os.path.join(output_folder, 'Expert.dat'), 'w') as f:
    #     json.dump(expert_data, f, indent=2)

    # expert_data == 0

    # expertp_data = generate_easy(9)
    # with open(os.path.join(output_folder, 'Expertplus.dat'), 'w') as f:
    #     json.dump(expertp_data, f, indent=2)

    # expertp_data == 0

    print(f'Exported!')


export_map()

shutil.make_archive('beatmap', 'zip', 'beatmap')

shutil.rmtree('beatmap')
