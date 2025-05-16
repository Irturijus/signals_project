import matplotlib.pyplot as plt
import json

# a script used for testing, to compare two beat signals of the same song (procuced using different parameters or algorithms)

with open("process_song/beats_array.json", "r") as file:
    beat_data = json.load(file)

beats_array = beat_data["beats"]

with open("process_song/beats_array2.json", "r") as file:
    beat_data2 = json.load(file)

beats_array2 = beat_data2["beats"]

plt.plot(beats_array)
plt.plot(beats_array2)
plt.show()