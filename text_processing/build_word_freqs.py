# This script builds a dictionary of word frequencies from one or more text files. 

import sys
import os
import re
import string
import json
import argparse
import random

NUM_WORD_FREQS = 1000
NUM_FILES_TO_PROCESS = 1000
ROOT_DIR = '/data/training_data/gutenberg/'
FILE_DESCRIPTIONS = ROOT_DIR + 'metadata/metadata.csv'
WORD_FREQ_FILE    = ROOT_DIR + 'metadata/word_freqs.csv'
INPUT_FILES_DIR   = ROOT_DIR + 'data/text/'

# FILE_DESCRIPTIONS is a csv file with the following format:

#          id,          title,   author,authoryearofbirth,authoryearofdeath,language,downloads,                                                         subjects,type
#     PG10000,The Magna Carta,Anonymous,,,                                    ['en'],      186,"{'Magna Carta', 'Constitutional history -- England -- Sources'}",

# Open FILE_DESCRIPTIONS and read in all the lines. For any English file, add its file id to a list of English files.
english_files = []
for ii, line in enumerate(open(FILE_DESCRIPTIONS)):
    if """['en']""" in line:
        line = line.split(',')
        file_id = line[0]
        english_files.append(file_id)


# Select a random subset of the files to process, but make sure we don't select the same file twice.
files_to_process = []
for ii in range(NUM_FILES_TO_PROCESS):
    file_id = random.choice(english_files)
    file_exists = os.path.isfile(INPUT_FILES_DIR + file_id + '_text.txt')
    while file_id in files_to_process or not file_exists:
        file_id = random.choice(english_files)
        file_exists = os.path.isfile(INPUT_FILES_DIR + file_id + '_text.txt')

    # Add the file id to the list of files to process.

    # Use the full path to the file and the file's actual name:
    file_name = INPUT_FILES_DIR + file_id + '_text.txt'
    files_to_process.append(file_name)

# Loop over all the files to process. Also keep track of the total
# number of words in all the files:
total_num_words = 0
word_freqs = {}
for file_name in files_to_process:
    # Open the file and read in all the lines:
    print('Processing file: {}'.format(file_name))
    for ii, line in enumerate(open(file_name)):
        # Remove punctuation and make all the letters lowercase:
        line = line.translate(str.maketrans('', '', string.punctuation)).lower()
        # Split the line into words:
        words = line.split()
        total_num_words += len(words)
        # Loop over all the words in the line:
        for word in words:
            # Add the word to the dictionary of word frequencies:
            if word in word_freqs:
                word_freqs[word] += 1
            else:
                word_freqs[word] = 1

# Convert the frequencies to probabilities:
for word in word_freqs:
    word_freqs[word] /= total_num_words

# Sort the dictionary of word frequencies by the number of times each word appears in the text:
sorted_word_freqs = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)

# Print out the top NUM_WORD_FREQS words.
# Also save them to a csv file (WORD_FREQS_FILE)
word_freqs_file = open(WORD_FREQ_FILE, 'w')
for ii in range(NUM_WORD_FREQS):
    # Print out the word and its frequency, using fixed width columns
    # for the word and the frequency:
    print('{:20s} {:10.6f}'.format(sorted_word_freqs[ii][0], sorted_word_freqs[ii][1]))
    word_freqs_file.write('{},{}\n'.format(sorted_word_freqs[ii][0], sorted_word_freqs[ii][1]))
word_freqs_file.close()

    

