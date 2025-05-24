import os
import shutil
import webbrowser
import time

if os.path.exists('process_song/beats_array.json'):
    os.remove('process_song/beats_array.json')

if os.path.exists('beatmap.zip'):
    os.remove('beatmap.zip')

if os.path.exists('beatmap'):
    shutil.rmtree('beatmap')

import marker
import convert_to_dat
import zipping

user_input = input('To check the result, a browser window will have to be opened. Do you agree (Y/N): ') 

if user_input == 'Y':
    print('When the website opens, press the folder icon on the website and select "beatmap.zip" as the input file, it is in the main directory.')
    time.sleep(5)
    webbrowser.open("https://allpoland.github.io/ArcViewer/")
else:
    print('To check the result, please search up arcviewer beat saber in your search engine of choice.')