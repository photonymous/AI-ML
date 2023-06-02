# Now that we have the word frequencies, we can use them to assign a "simplicity" score to each file.
# The simplicity score is the weighted sum of the frequencies of the words in the file, weighted by
# the background word frequencies previously calculated. There are over 10,000 files to process, so
# we should use multiprocessing to make the most of our 24 cores.

import multiprocessing
import os
import string

ROOT_DIR = '/data/training_data/gutenberg/'
WORD_FREQ_FILE    = ROOT_DIR + 'metadata/word_freqs.csv'
INPUT_FILES_DIR   = ROOT_DIR + 'data/text/'
FILE_DESCRIPTIONS = ROOT_DIR + 'metadata/metadata.csv'
SIMPLICITY_SCORES = ROOT_DIR + 'metadata/simplicity_scores_sorted.csv'
NUM_WORKERS = 48

# Open FILE_DESCRIPTIONS and read in all the lines. For any English file, add its file id to a list of English files.
english_files = []
for ii, line in enumerate(open(FILE_DESCRIPTIONS)):    
    if """['en']""" in line:
        line = line.split(',')
        file_id = line[0]
        file_full_path = INPUT_FILES_DIR + file_id + '_text.txt'
        file_exists = os.path.isfile(file_full_path)
        if file_exists:
            english_files.append(file_full_path)

# Read in the word frequencies from word_freqs.csv. These will be our "weights"
# when we calculate the simplicity scores for each file:
word_freqs = {}
for ii, line in enumerate(open(WORD_FREQ_FILE)):
    line = line.split(',')
    word = line[0]
    freq = float(line[1])
    word_freqs[word] = freq



def process_file(file_name, word_freqs, job_num):
    # Open the file and read in all the lines:
    file_word_freqs = {}
    for ii, line in enumerate(open(file_name)):
        # Remove punctuation and make all the letters lowercase:
        line = line.translate(str.maketrans('', '', string.punctuation)).lower()
        # Split the line into words:
        words = line.split()
        # Loop over all the words in the line:
        for word in words:
            # Add the word to the dictionary of word frequencies:
            if word in file_word_freqs:
                file_word_freqs[word] += 1
            else:
                file_word_freqs[word] = 1

    # Convert the frequencies to probabilities:
    total_num_words = sum(file_word_freqs.values())
    for word in file_word_freqs:
        file_word_freqs[word] /= total_num_words

    # Calculate the simplicity score for this file:
    simplicity_score = 0
    for word in file_word_freqs:
        if word in word_freqs:
            simplicity_score += file_word_freqs[word] * word_freqs[word]

    # format an integer with 6 digits: 
    print('Simplicity Score: {:0.6f}, Job Number: {}'.format(simplicity_score, job_num), flush=True)

    # Return the simplicity score:
    return simplicity_score

# # Lets try it without workers first:
# simplicity_scores = []
# for ii, file_full_name in enumerate(english_files):
#     simplicity_scores.append(process_file(file_full_name, word_freqs))



# Create a pool of workers:
pool = multiprocessing.Pool(NUM_WORKERS)

# Create a list of arguments to pass to process_file():
args = []
for ii, file_full_name in enumerate(english_files):
    args.append((file_full_name, word_freqs, ii))

# Run the jobs in parallel:
simplicity_scores = pool.starmap(process_file, args)

# Close the pool of workers:
pool.close()

# Sort the files by simplicity score, from highest to lowest:
sorted_scores = sorted(zip(simplicity_scores, english_files), reverse=True)

# Save the sorted data to a file:
with open(SIMPLICITY_SCORES, 'w') as f:
    for (simplicity_score, file_full_name) in sorted_scores:
        f.write('{},{}\n'.format(file_full_name, simplicity_score))


