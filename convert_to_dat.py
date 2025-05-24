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

    # Whether last block was a directional block
    dir_status = {'X': False, 'Y': False}
    # Keeps track of last direction [True, True] = [Up(0), Right(3)]
    track_dir = {'X': True, 'Y': True}

    prev_loc = set()

    for i, val in enumerate(beats):
        if not val:
            continue

        cur_loc = set()

        second = i / samplerate
        beat = seconds_to_beat(second)

        # Stack generator
        if val > (0.75-(difficulty-1)*0.01) * max_val:
            note_count = 3
        elif val > (0.5-(difficulty-1)*0.01) * max_val:
            note_count = 2
        else:
            note_count = 1

        # Direction generator, a huge mess to implement logic
        direction = 8
        if note_count in [2, 3]:  # Separate case for 3 notes to be only up down or any
            if dir_status['Y']:
                track_dir['Y'] = not track_dir['Y']
                direction = 0 if track_dir['Y'] else 1
                dir_status['Y'] = False
            else:
                direction = random.choice([0, 1, 8])
                if direction != 8:
                    dir_status['Y'] = True
        else:
            if dir_status['X']:
                track_dir['X'] = not track_dir['X']
                direction = 3 if track_dir['X'] else 2
                dir_status['X'] = False
            elif dir_status['Y']:
                track_dir['Y'] = not track_dir['Y']
                direction = 0 if track_dir['Y'] else 1
                dir_status['Y'] = False
            else:
                if random.random() < 0.6:
                    direction = 8
                else:
                    direction = random.randint(0, 7)
                    if direction in [2, 3]:
                        dir_status['X'] = True
                        track_dir['X'] = (direction == 3)
                    if direction in [0, 1]:
                        dir_status['Y'] = True
                        track_dir['Y'] = (direction == 0)

        x = random.randint(0, 3)
        if x == 0:
            color = 0
        elif x == 3:
            color = 1
        else:
            color = random.randint(0, 1)

        for _ in range(note_count):
            while True:  # Generate positions that aren't same as last
                available_y = {0, 1, 2}

                if not available_y:
                    break
                y = random.choice(list(available_y))
                available_y.remove(y)

                loc = (x, y)
                if loc not in prev_loc:
                    cur_loc.add(loc)
                    break

            color_notes.append({
                'b': round(beat, 3),  # Beat
                'x': x,  # Line Index
                'y': y,  # Line Layer
                'a': 0,
                'c': color,
                'd': direction,  # Cut direction
            })
        prev_loc = cur_loc

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