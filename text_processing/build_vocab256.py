
import numpy as np

VOCAB_OUTPUT_FILE = "/data/training_data/vocab256.ascii"

# create an empty vocab list:
vocab = []

# append the horizontal tab character to the vocab list. Do it as a hex character:
idx = 0
vocab.append('\x09')
# append the line feed character to the vocab list. Do it as a hex character:
idx += 1
vocab.append('\x0A')
# append the carriage return character to the vocab list. Do it as a hex character:
idx += 1
vocab.append('\x0D')

# Now add the printable characters (32-126) to the next vocab positions:1
for i in range(32, 127):
    idx += 1
    vocab.append(chr(i))

# Number of vocab entries remaining:
remaining = 256 - idx - 1


common_words = ["the","and","that","was","his","with","for","had","not","but","you","which","from","this","her","have","they","were","all","are","him","she","their","one","there","been","when","who","them","would","said","will","its","more","into","has","what","out","some","then","very","time","than","could","other","now","may","about","only","man","upon","any","our","these","your","two","little","after","made","great","should","over","like","can","see","such","before"]

common_endings = ["ing","ed","tion","es","ous","ment","ness","ence","ance","able","ful","able","ible","al","ial","en","er","est","ic","ing","ion","tion","ty","ive","less","ly","ment","ness","ous"]

common_beginnings = ["an","ante","anti","auto","circum","co","com","con","de","dis","en","ex","extra","hetero","homo","homeo","hyper","il","im","in","ir","in","inter","intra","intro","macro","micro","mono","non","omni","post","pre","pro","sub","sym","syn","tele","trans","tri","un","uni","up"]

common_bigrams = ["th","he","in","er","an","re","nd","on","en","at","ou","ed","ha","to","or","it","is","hi","es","ng","th","he","in","er","an","re","on","at","en","nd","ti","es","or","te","of","ed","is","it","al","ar","st","to","we","ou","ea","ng","ha","as","ia","li","le","ne","hi","me","de","co","ta","ec","si","ll","ve","ss","se","am","ro"]

common_trigrams = ["the","and","ing","her","hat","his","tha","ere","for","ent","ion","ter","was","you","ith","ver","all","wit","thi","tio","the","and","tha","ent","ing","ion","tio","for","nde","has","nce","edt","tis","oft","sth","men"]


# Go through each list and remove any duplicate entries. Use set() for this:
def dedup(input):
    # create a set form the input:
    output = set(input)
    # create a list from the set:
    output = list(output)

    return output

def dedup_multi(input1, input2):
    output = []
    # build up output from only the unique values in input1 that aren't in input2:
    for item in input1:
        if item not in input2:
            output.append(item)
    return output

def count_occurances(input, corpus):
    # Build up a dictionary of the number of occurances of each item in input found in the corpus, which
    # is a single long string:
    output = []
    for item in input:
        output.append(corpus.count(item))
    return output

common_words      = dedup(common_words)
common_endings    = dedup(common_endings)
common_beginnings = dedup(common_beginnings)
common_bigrams    = dedup(common_bigrams)
common_trigrams   = dedup(common_trigrams)

# Now dedup the lists with respect to each other:
common_trigrams   = dedup_multi(common_trigrams,   common_words + common_endings + common_beginnings + common_bigrams                  )
common_bigrams    = dedup_multi(common_bigrams,    common_words + common_endings + common_beginnings +                  common_trigrams)
common_beginnings = dedup_multi(common_beginnings, common_words + common_endings +                     common_bigrams + common_trigrams)
common_endings    = dedup_multi(common_endings,    common_words +                  common_beginnings + common_bigrams + common_trigrams)
common_words      = dedup_multi(common_words,                     common_endings + common_beginnings + common_bigrams + common_trigrams)

    

# Print total length of all each list:
print("Number of common words: ", len(common_words))
print("Number of common endings: ", len(common_endings))
print("Number of common beginnings: ", len(common_beginnings))
print("Number of common bigrams: ", len(common_bigrams))
print("Number of common trigrams: ", len(common_trigrams))

print("Total length of all lists: ", len(common_words) + len(common_endings) + len(common_beginnings) + len(common_bigrams) + len(common_trigrams))
print("Remaining vocab entries: ", remaining)


reference_corpus = "/data/training_data/gutenberg/data/english_corpus.ascii"

# Score the list elements by their frequency in the corpus.
# First, get a list of 10,000 random lines from the corpus.
the_corpus = open(reference_corpus, "r")
# How many bytes are in the file?
file_size = the_corpus.seek(0, 2)
concatenated_lines = ""
for ii in range(10000):
    # Using the file size, jump to a random position in the file:
    the_corpus.seek(np.random.randint(file_size - 1000))
    # from there, find the next newline character:
    the_corpus.readline()
    # now read a full line:
    concatenated_lines += the_corpus.readline()
    print("Read line ", ii, end="\r")

common_words_counts = count_occurances(common_words, concatenated_lines)
common_endings_counts = count_occurances(common_endings, concatenated_lines)
common_beginnings_counts = count_occurances(common_beginnings, concatenated_lines)
common_bigrams_counts = count_occurances(common_bigrams, concatenated_lines)
common_trigrams_counts = count_occurances(common_trigrams, concatenated_lines)

# Now, sort the lists by their counts. Each of these should be a tuple containing
# the sorted words and the sorted counts:
common_words_sorted = sorted(zip(common_words, common_words_counts), key=lambda x: x[1], reverse=True)
common_endings_sorted = sorted(zip(common_endings, common_endings_counts), key=lambda x: x[1], reverse=True)
common_beginnings_sorted = sorted(zip(common_beginnings, common_beginnings_counts), key=lambda x: x[1], reverse=True)
common_bigrams_sorted = sorted(zip(common_bigrams, common_bigrams_counts), key=lambda x: x[1], reverse=True)
common_trigrams_sorted = sorted(zip(common_trigrams, common_trigrams_counts), key=lambda x: x[1], reverse=True)


# Now print out all of the sorted lists, showing the item and its count:
print("\nCommon words:")
for ii in range(len(common_words_sorted)):
    print(common_words_sorted[ii][0], common_words_sorted[ii][1])

print("\nCommon endings:")
for ii in range(len(common_endings_sorted)):
    print(common_endings_sorted[ii][0], common_endings_sorted[ii][1])

print("\nCommon beginnings:")
for ii in range(len(common_beginnings_sorted)):
    print(common_beginnings_sorted[ii][0], common_beginnings_sorted[ii][1])

print("\nCommon bigrams:")
for ii in range(len(common_bigrams_sorted)):
    print(common_bigrams_sorted[ii][0], common_bigrams_sorted[ii][1])

print("\nCommon trigrams:")
for ii in range(len(common_trigrams_sorted)):
    print(common_trigrams_sorted[ii][0], common_trigrams_sorted[ii][1])


# Now build two big lists. One is a list of all the items in the sorted lists, and the other is a list of
# the counts of those items. Then sort these new lists by the counts, and print them out:
all_items = common_words + common_endings + common_beginnings + common_bigrams + common_trigrams
all_counts = common_words_counts + common_endings_counts + common_beginnings_counts + common_bigrams_counts + common_trigrams_counts

# Create all_scores, which converts the counts to scores. This is done by
# multiplying the count by the length of the item:
all_scores = [all_counts[ii] * len(all_items[ii]) for ii in range(len(all_items))]


# Now sort the lists by the scores. Create two new sorted lists. Do not use tuples:
all_items_sorted = [x for _,x in sorted(zip(all_scores, all_items), key=lambda x: x[0], reverse=True)]
all_scores_sorted = sorted(all_scores, reverse=True)



# Prune the list down to just the items with the top scores. Keep "remaining" items:
pruned_items_sorted = all_items_sorted[:remaining]
pruned_scores_sorted = all_scores_sorted[:remaining]


# Now print out all of the sorted lists, showing the item and its count:
print("\npruned items:")
for ii in range(len(pruned_items_sorted)):
    print(pruned_items_sorted[ii], pruned_scores_sorted[ii])


# Add the pruned items to the vocab list:
vocab += pruned_items_sorted

# Print the final vocab list, with the index of each item:
print("\nFinal vocab list:")
for ii in range(len(vocab)):
    print(ii, vocab[ii])


# Save the vocab by just printing one item per line. For line feed and carriage return, use
# their hex values: 0x0A and 0x0D. We'll interpret them when we read the file back in.
with open(VOCAB_OUTPUT_FILE, "w") as f:
    for ii in range(len(vocab)):
        if vocab[ii] == "\n":
            f.write("0x0A\n")
        elif vocab[ii] == "\r":
            f.write("0x0D\n")
        else:
            f.write(vocab[ii] + "\n")


print("Vocab file created: ", VOCAB_OUTPUT_FILE)


