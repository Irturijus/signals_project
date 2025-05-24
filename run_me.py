import os
import shutil
import webbrowser
import time

print('To check the result, a browser window will have to be opened')

if os.path.exists('process_song/beats_array.json'):
    os.remove('process_song/beats_array.json')

if os.path.exists('beatmap.zip'):
    os.remove('beatmap.zip')

if os.path.exists('beatmap'):
    shutil.rmtree('beatmap')

import marker
import convert_to_dat
import zipping

webbrowser.open("https://allpoland.github.io/ArcViewer/")