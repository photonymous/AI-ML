#!/usr/bin/env python3

# This file can be used as an executable at the command line to 
# tokenize an input .ascii text file on stdin, and output the
# tokenized version on stdout. The .ascii files should be
# created by processing utf-8 .txt files using utf8_to_ascii.py.
# Such utf-8 .txt files include Gutenberg data, and the
# TinyStories.
#
# Usage:
#   tokenizer.py VOCAB_LEN < input_file.ascii > output_file.tokens
#
# Where VOCAB_LEN is the number of tokens in the vocabulary.
# You must provide VOCAB_LEN as the first argument, and it can be
# 256, 512 or 1024. The output token ids will be stored in uint8
# if VOCAB_LEN is 256, and uint16 if VOCAB_LEN is 512 or 1024.
#
# Note that this script selectively loads vocab files from the
# /data/training_data/ directory, depending on the VOCAB_LEN.
# These files were created with "build_vocab.py" by processing
# the full Gutenberg english_corpus.ascii file (18GB).
#   

# TODO: Since this is line based, we should parallelize this
#       using multiprocessing so we can use all our cores.

import sys
from tokenizers import Tokenizer
import numpy as np
import multiprocessing as mp

input_file = sys.stdin
output_file = sys.stdout.buffer

# Check if there are more than 1 argument, if not, exit and print usage:
if len(sys.argv) < 2:
    print("Usage: {} VOCAB_LEN < input_file.ascii > output_file.tokens. \n VOCAB_LEN can be 256, 512 or 1024".format(sys.argv[0]))
    sys.exit(1)

# Grab the first argument as the vocab length:
vocab_len = int(sys.argv[1])

output_data_type = 'uint8' if vocab_len == 256 else 'uint16'
vocab_file = "/data/training_data/vocab{}.json".format(vocab_len)

tokenizer = Tokenizer.from_file(vocab_file)
  


# read and tokenize the input file:
while True:
    # Read a line from stdin and include the newline character and possibly linefeed in the line:
    line = input_file.readline()
    # If we get an empty string, we have reached the end of the input:
    if not line:
        break
    # Tokenize the line:
    tokens = tokenizer.encode(line)
    # Write the tokens to stdout as binary data. 
    # Ensure the tokens.ids are uint8 or uint16, depending on output_data_type.:
    # Make a numpy array from the tokens.ids:
    tokens = np.array(tokens.ids, dtype=output_data_type)
    # Write the np.array() to stdout:
    output_file.write(tokens)

    # Write a newline character to stdout:
    # OOPS. Why did we have this line? 
    # output_file.write(b'\n')
    
# Flush the output buffer:
output_file.flush()



