# This script takes a text file as input and outputs a histogram of the
# frequencies of each character in the file. It uses numpy and matplotlib.
# It processes the file in chunks of 1MB at a time and uses vectorization.

import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "/data/training_data/gutenberg_corpus_21MB.ascii"
#INPUT_FILE = "/data/training_data/TinyStories-valid.txt"

CHUNK_SIZE = 2**10

histogram_counts = np.zeros(256, dtype=np.int64)

# Open the file for reading in binary mode:
the_file = open(INPUT_FILE, "rb")

if not the_file:
    print("Error opening file: " + INPUT_FILE)
    exit(1)

# Read and process the file in chunks of 1MB at a time:
keep_going = True
while keep_going:
    chunk = the_file.read(CHUNK_SIZE)
    if chunk:
        # Convert the chunk to a numpy array of uint8:
        chunk_array = np.frombuffer(chunk, dtype=np.uint8)
        # cast it as int64:
        chunk_array = chunk_array.astype(np.int64)
        # Count the number of occurrences of each byte value:
        histogram_counts += np.bincount(chunk_array, minlength=256)

        # # Find the locations of any characters that are > 127:
        # locations = np.where(chunk_array > 127)[0]

        # if len(locations) > 0:
        #     # for every occurance of a character > 127, print the context:
        #     for i in range(len(locations)):
        #         # Print the context for the first instance in this chunk:
        #         location = locations[i]
        #         print("<{}> :".format(chunk[location]), end="")
        #         # Print the context:
        #         for j in range(location-10, location+10):
        #             if j < 0 or j >= len(chunk):
        #                 continue
        #             if chunk_array[j] > 127:
        #                 print("<{}>".format(chunk_array[j]), end="")
        #             else:
        #                 print("{}".format(chr(chunk_array[j])), end="")
        #         print()
    else:
        keep_going = False

    

# Close the file:
the_file.close()

# Print the histogram:
for i in range(256):
    print("{:3d}: {}".format(i, histogram_counts[i]))

# Print the number of non-zero counts:
non_zero_counts = np.count_nonzero(histogram_counts)
print("Number of non-zero counts: {}".format(non_zero_counts))

# Print the number of zero counts:
zero_counts = 256 - non_zero_counts
print("Number of zero counts: {}".format(zero_counts))
