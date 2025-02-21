import numpy as np
import os

# Load the .npz file
data = np.load("data/mini_weather_hmm.npz")

# List all arrays stored in the file
print(type(data), data.files)

data2 = np.load("data/mini_weather_sequences.npz")

print(type(data2), data2.files)

data3 = np.load("data/full_weather_hmm.npz")

print(type(data3), data3)

data4 = np.load("data/full_weather_sequences.npz")

print(type(data4['best_hidden_state_sequence']), data4)