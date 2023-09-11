import numpy as np
import pandas as pd

# Constants
MIN_RANGE_UP = 0.01
SOURCE_FILE = '../data/spy.csv.gz'
DESTINATION_FILE = './spy.target.csv.gz'

# Loading data
df = pd.read_csv(SOURCE_FILE, compression='gzip')

# Initialize variables
values = df['close']
length = len(values)
target = np.zeros(length)

# Algorithm to generate target
i = 0
while i < length:

    curr_value = values[i]
    prev_value = curr_value

    open_idx = None
    close_idx = None

    for j in range(i + 1, length):
        next_value = values[j]
        ratio = next_value / curr_value - 1

        if open_idx is not None: # Look for position close
            if next_value < prev_value:
                close_idx = j - 1
                i = close_idx
                break
            prev_value = next_value
        elif ratio < 0:
            break
        elif ratio >= MIN_RANGE_UP: # Look for position open
            open_idx = i
            prev_value = next_value

    if open_idx is not None and close_idx is not None:
        target[open_idx] = 1
        target[close_idx] = -1

    i += 1

# Assign target to dataset
df['target'] = target.astype('int')

# Generate the new file
df.to_csv(DESTINATION_FILE, compression='gzip', index=False)