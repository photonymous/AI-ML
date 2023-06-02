import matplotlib.pyplot as plt
import numpy as np
import math


ROOT_DIR = '/data/training_data/gutenberg/'
WORD_FREQ_FILE    = ROOT_DIR + 'metadata/word_freqs.csv' 
INPUT_FILES_DIR   = ROOT_DIR + 'data/text/'
FILE_DESCRIPTIONS = ROOT_DIR + 'metadata/metadata.csv'
SIMPLICITY_SCORES = ROOT_DIR + 'metadata/simplicity_scores_sorted.csv'

TOP_N_WORDS = 50

# Read in the simplicity scores from simplicity_scores_sorted.csv:
simplicity_scores = {}
for ii, line in enumerate(open(SIMPLICITY_SCORES)):
    line = line.split(',')
    file_name = line[0]
    simplicity_score = float(line[1])
    simplicity_scores[file_name] = simplicity_score

# Plot the histogram of the simplicity scores. These values range from roughly 0 to 0.012,
# and lets do it as a PDF so that the area under the curve is roughly 1.

# lets compute the pdf manually by calling hist() and getting the counts, then normalizing:
n, bins, patches = plt.hist(simplicity_scores.values(), bins=100, density=False)

# normalize the bin counts so that the area under the curve is 1:
plt.figure(1)
plt.clf()
plt.bar(bins[:-1], n/sum(n), width=(bins[1]-bins[0]))
plt.xlabel('Simplicity Score')
plt.ylabel('Probability Density')
plt.title('Histogram of Simplicity Scores of Gutenberg English Books')

plt.grid(True)
plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=20)


# Plot the CDF in a new figure, using the normalized bin counts from the histogram:
plt.figure(2)
plt.clf()
plt.bar(bins[:-1], n.cumsum()/sum(n), width=(bins[1]-bins[0]))
plt.xlabel('Simplicity Score')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Simplicity Scores of Gutenberg English Books')
plt.grid(True)
plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=20)

# Plot the word frequencies:
gutenberg_word_freqs = {}
for ii, line in enumerate(open(WORD_FREQ_FILE)):
    line = line.split(',')
    word = line[0]
    freq = float(line[1])
    gutenberg_word_freqs[word] = freq
# Plot the word frequencies:
plt.figure(3)
plt.clf()
plt.semilogy(gutenberg_word_freqs.values())
plt.xlabel('Word Rank')
plt.ylabel('Word Frequency')
plt.title('Gutenberg Top {} Word Frequencies'.format(len(gutenberg_word_freqs)))
plt.grid(True)

# Zoom in on just the TOP_N_WORDS, and show the words on the x axis:
plt.figure(4)
plt.clf()
plt.semilogy(range(TOP_N_WORDS), list(gutenberg_word_freqs.values())[:TOP_N_WORDS])
plt.xlabel('Word')
plt.ylabel('Word Frequency')
plt.title('Gutenberg Top {} Word Frequencies'.format(TOP_N_WORDS))
plt.xticks(range(TOP_N_WORDS), list(gutenberg_word_freqs.keys())[:TOP_N_WORDS], rotation='vertical')

# Get the 10^x exponent for ymin and ymax:
ymin, ymax = plt.ylim()
ymin_exp = np.log10(ymin)
ymax_exp = np.log10(ymax)
# Round ymin_exp down using floor, and round ymax_exp up using ceil:
ymin_exp = math.floor(ymin_exp)
ymax_exp = math.ceil(ymax_exp)
# Set the y axis limits to be 10^ymin_exp and 10^ymax_exp:
plt.ylim([10**ymin_exp, 10**ymax_exp])
# Turn minorticks_on, but only for the y axis:
plt.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.grid(True, which='minor', axis='y')
plt.grid(True, which='major', axis='x')

# What simplicity score corresponds to the 5th percentile and the 95th percentile?
# Also find the mean and median.
# Use the sorted simplicity score data to find these values:
simplicity_scores_sorted = sorted(simplicity_scores.items(), key=lambda x: x[1], reverse=False)
# Get the 5th percentile and the 95th percentile:
percentile_5 = simplicity_scores_sorted[int(0.05*len(simplicity_scores_sorted))][1]
percentile_95 = simplicity_scores_sorted[int(0.95*len(simplicity_scores_sorted))][1]
mean_simplicity = np.mean([x[1] for x in simplicity_scores_sorted])
median_simplicity = np.median([x[1] for x in simplicity_scores_sorted])
print(' 5th percentile simplicity: {:.6f}'.format(percentile_5))
print('95th percentile simplicity: {:.6f}'.format(percentile_95))
print('Mean simplicity:            {:.6f}'.format(mean_simplicity))
print('Median simplicity:          {:.6f}'.format(median_simplicity))

# Show all the plots:
plt.show()


