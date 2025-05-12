import matplotlib.pyplot as plt
import json

with open("process_song/beats_array.json", "r") as file:
    beat_data = json.load(file)

beats_array = beat_data["beats"]

with open("process_song/beats_array.json", "r") as file:
    beat_data2 = json.load(file)

beats_array2 = beat_data["beats"]

#plt.plot(beats_array)
plt.plot(beats_array2)
plt.show()