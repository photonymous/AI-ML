from tokenizers import Tokenizer, processors, decoders
from tokenizers.decoders import BPEDecoder, ByteLevel
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, PreTokenizer, Sequence

VOCAB_SIZE = 256

CORPUS_FILE = "/data/training_data/gutenberg_corpus_21MB.txt"
#CORPUS_FILE = "/data/training_data/gutenberg/data/english_corpus.ascii"

VOCAB_OUTPUT_FILE = "/data/training_data/vocab{}.json".format(VOCAB_SIZE)


# Initialize a tokenizer
tokenizer = Tokenizer(BPE())

# Use the byte-level pre-tokenizer
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# Initialize a trainer
trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=VOCAB_SIZE)

# Train the tokenizer
tokenizer.model = BPE()
tokenizer.train([CORPUS_FILE], trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False, add_prefix_space=False)

# Now you can encode text
text_to_encode = "Hello, World! Man, life is great. I love tabs, " + '\x09' + "line feeds, and carriage returns." + '\x0A' + '\x0D' + "I also love       spaces."
output = tokenizer.encode(text_to_encode)
print(output.tokens)


# Decod the output and print it:
tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)
decoded = tokenizer.decode(output.ids)
print("decoded:  ", decoded)
print("original: ", text_to_encode)

# # print entire vocabulary and their index:
# for i in range(0,VOCAB_SIZE):
#     print(i, tokenizer.decode([i]))


tokenizer.save(VOCAB_OUTPUT_FILE)

# # How to load:
# from tokenizers import Tokenizer
# tokenizer = Tokenizer.from_file("tokenizer.json")












# import numpy as np
# import string

# VOCAB_OUTPUT_FILE = "/data/training_data/vocab256.ascii"

# # create an empty vocab list:
# vocab = []

# # append the horizontal tab character to the vocab list. Do it as a hex character:
# idx = 0
# vocab.append('\x09')
# # append the line feed character to the vocab list. Do it as a hex character:
# idx += 1
# vocab.append('\x0A')
# # append the carriage return character to the vocab list. Do it as a hex character:
# idx += 1
# vocab.append('\x0D')

# # Now add the printable characters (32-126) to the next vocab positions:1
# for i in range(32, 127):
#     idx += 1
#     vocab.append(chr(i))

# # Number of vocab entries remaining:
# remaining = 256 - idx - 1

# # Note that we'll be adding "the", "and", and "ing" manually

# common_words = ["that","was","his","with","for","had","not","but","you","which","from","this","her","have","they","were","all","are","him","she","their","one","there","been","when","who","them","would","said","will","its","more","into","has","what","out","some","then","very","time","than","could","other","now","may","about","only","man","upon","any","our","these","your","two","little","after","made","great","should","over","like","can","see","such","before"]

# common_endings = ["ed","tion","es","ous","ment","ness","ence","ance","able","ful","able","ible","al","ial","en","er","est","ic","ion","tion","ty","ive","less","ly","ment","ness","ous"]

# common_beginnings = ["an","ante","anti","auto","circum","co","com","con","de","dis","en","ex","extra","hetero","homo","homeo","hyper","il","im","in","ir","in","inter","intra","intro","macro","micro","mono","non","omni","post","pre","pro","sub","sym","syn","tele","trans","tri","un","uni","up"]

# # generate all possible combinations of two lowercase characters:
# common_bigrams = [a+b for a in string.ascii_lowercase for b in string.ascii_lowercase]

# common_trigrams = ["tha","eth","for","est","hes","oth","tth","wit","res","rth","you","edt","ast","con","nce","man","out","she","eri","att","hin","ine","rin","han","ent","nth","all","tio","ith","fth","thi","ate","ont","was","sin","eof","ons","era","ted","hen","ort","ant","ein","eco","hea","ran","ill","com","ere","hat","sth","his","ers","dth","rea","are","tin","sof","sto","eve","din","sta","ght","not","esa","ngt","ndt","ave","ive","hec","nde","igh","her","ion","int","ter","oft","ati","ver","san","ear","ess","ean","ist","one","ome","our","hem","ore","ert","edi","nto","men","eda","tan","tho","ain"]

# common_quadrigrams = ["that","ther","with","tion","here","ould","ight","have","hich","whic","this","thin","they","atio","ever","from","ough","were","hing","ment"]



# # Go through each list and remove any duplicate entries. Use set() for this:
# def dedup(input):
#     # create a set form the input:
#     output = set(input)
#     # create a list from the set:
#     output = list(output)

#     return output

# def dedup_multi(input1, input2):
#     output = []
#     # build up output from only the unique values in input1 that aren't in input2:
#     for item in input1:
#         if item not in input2:
#             output.append(item)
#     return output

# def count_occurances(input, corpus):
#     # Build up a dictionary of the number of occurances of each item in input found in the corpus, which
#     # is a single long string:
#     output = []
#     for item in input:
#         output.append(corpus.count(item))
#     return output

# common_words      = dedup(common_words)
# common_endings    = dedup(common_endings)
# common_beginnings = dedup(common_beginnings)
# common_bigrams    = dedup(common_bigrams)
# common_trigrams   = dedup(common_trigrams)
# common_quadrigrams= dedup(common_quadrigrams)

# # Now dedup the lists with respect to each other:
# common_quadrigrams= dedup_multi(common_quadrigrams,                     common_words + common_endings + common_beginnings + common_bigrams + common_trigrams)
# common_trigrams   = dedup_multi(common_trigrams,   common_quadrigrams + common_words + common_endings + common_beginnings + common_bigrams                  )
# common_bigrams    = dedup_multi(common_bigrams,    common_quadrigrams + common_words + common_endings + common_beginnings +                  common_trigrams)
# common_beginnings = dedup_multi(common_beginnings, common_quadrigrams + common_words + common_endings +                     common_bigrams + common_trigrams)
# common_endings    = dedup_multi(common_endings,    common_quadrigrams + common_words +                  common_beginnings + common_bigrams + common_trigrams)
# common_words      = dedup_multi(common_words,      common_quadrigrams +                common_endings + common_beginnings + common_bigrams + common_trigrams)

    

# # Print total length of all each list:
# print("Number of common words: ", len(common_words))
# print("Number of common endings: ", len(common_endings))
# print("Number of common beginnings: ", len(common_beginnings))
# print("Number of common bigrams: ", len(common_bigrams))
# print("Number of common trigrams: ", len(common_trigrams))
# print("Number of common quadrigrams: ", len(common_quadrigrams))

# print("Total length of all lists: ", len(common_words) + len(common_endings) + len(common_beginnings) + len(common_bigrams) + len(common_trigrams) + len(common_quadrigrams))
# print("Remaining vocab entries: ", remaining)



# # Concatentate all the lists into one big list:
# candidate_tokens = common_words + common_endings + common_beginnings + common_bigrams + common_trigrams + common_quadrigrams

# # Get a list of 100,000 random lines fromt the corpus
# reference_corpus = "/data/training_data/gutenberg/data/english_corpus.ascii"
# the_corpus = open(reference_corpus, "r")
# # How many bytes are in the file?
# file_size = the_corpus.seek(0, 2)
# concatenated_lines = ""
# for ii in range(100000):
#     # Using the file size, jump to a random position in the file:
#     the_corpus.seek(np.random.randint(file_size - 1000))
#     # from there, find the next newline character:
#     the_corpus.readline()
#     # now read a full line:
#     concatenated_lines += the_corpus.readline()
#     print("Read line ", ii, end="\r")
# print()

# # Score the candidate tokens by multiplying their length-1 times their frequency in the corpus:
# candidate_token_scores = count_occurances(candidate_tokens, concatenated_lines)
# candidate_token_scores = [(len(item)-1) * candidate_token_scores[ii] for ii, item in enumerate(candidate_tokens)]

# # Sort the candidate tokens by their score:
# candidate_token_scores, candidate_tokens = zip(*sorted(zip(candidate_token_scores, candidate_tokens), reverse=True))

# # Print them and their scores:
# for ii in range(len(candidate_tokens)):
#     print(candidate_tokens[ii], candidate_token_scores[ii])

# # Manually add "the", "and", and "ing" to the vocab, and remove them from the concatenated lines:
# vocab.append("the")
# vocab.append("and")
# vocab.append("ing")
# concatenated_lines = concatenated_lines.replace("the", " ")
# concatenated_lines = concatenated_lines.replace("and", " ")
# concatenated_lines = concatenated_lines.replace("ing", " ")

# # Loop over the candidate tokens from highest to lowest score. For each one, first add it to the vocab.
# # Then remove it from the concatenated_lines, and score the remaining tokens. Sort them, and repeat
# # until the vocab is full:
# while len(vocab) < 256:
#     # Add the highest scoring candidate token to the vocab:
#     vocab.append(candidate_tokens[0])

#     # Print it and its score:
#     print("Added ", candidate_tokens[0], " with score ", candidate_token_scores[0], "Vocab size: ", len(vocab))
#     score = candidate_token_scores[0]
#     if score == 0:
#        print("WTF")

#     # Remove it from the concatenated lines:
#     concatenated_lines = concatenated_lines.replace(candidate_tokens[0], " ")
#     # Remove it from the candidate tokens:
#     candidate_tokens = candidate_tokens[1:]
#     # Score the remaining candidate tokens:
#     candidate_token_scores = count_occurances(candidate_tokens, concatenated_lines)
#     candidate_token_scores = [(len(item)-1) * candidate_token_scores[ii] for ii, item in enumerate(candidate_tokens)]
#     # Sort the candidate tokens by their score:
#     candidate_token_scores, candidate_tokens = zip(*sorted(zip(candidate_token_scores, candidate_tokens), reverse=True))

#     print("Vocab size: ", len(vocab), end="\r") 

# # Print the final vocab list, with the index of each item:
# print("\nFinal vocab list:")
# for ii in range(len(vocab)):
#     print(ii, vocab[ii])

# # Save the vocab by just printing one item per line. For line feed and carriage return, use
# # their hex values: 0x0A and 0x0D. We'll interpret them when we read the file back in.
# with open(VOCAB_OUTPUT_FILE, "w") as f:
#     for ii in range(len(vocab)):
#         if vocab[ii] == "\n":
#             f.write("0x0A\n")
#         elif vocab[ii] == "\r":
#             f.write("0x0D\n")
#         else:
#             f.write(vocab[ii] + "\n")


# print("Vocab file created: ", VOCAB_OUTPUT_FILE)
