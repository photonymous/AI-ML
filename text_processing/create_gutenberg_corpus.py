# Build a singular corpus from the English Gutenberg texts, that is in order from
# highest simplicity score to lowest simplicity score.

LOWER_SIMPLICITY = 0.007383
UPPER_SIMPLICITY = 0.011794

ROOT_DIR = '/data/training_data/gutenberg/'
INPUT_FILES_DIR            = ROOT_DIR + 'data/text/'
SIMPLICITY_SCORES          = ROOT_DIR + 'metadata/simplicity_scores_sorted.csv'
ENGLISH_CORPUS_OUTPUT_FILE = ROOT_DIR + 'data/english_corpus.txt'

# Iterate over all of the text files referenced in the simplicity scores file.
# If their simplicity score is between LOWER_SIMPLICITY and UPPER_SIMPLICITY,
# then append their text to the English corpus file.
with open(ENGLISH_CORPUS_OUTPUT_FILE, 'w') as f:
    for ii, line in enumerate(open(SIMPLICITY_SCORES)):
        line = line.split(',')
        full_file_name = line[0]
        simplicity_score = float(line[1])
        if simplicity_score >= LOWER_SIMPLICITY and simplicity_score <= UPPER_SIMPLICITY:
            print('Processing file: {} {} {:.6f}'.format(full_file_name, ii, simplicity_score), flush=True)
            for line in open(full_file_name):
                f.write(line)
