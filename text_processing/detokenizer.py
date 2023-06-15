#!/usr/bin/env python3

# This file can be used as an executable at the command line to 
# detokenize an input .token file on stdin, and output the
# detokenized (ascii) version on stdout. The .token files should be
# created by processing .ascii files using tokenizer.py.
#
# Usage:
#   detokenizer.py VOCAB_LEN < input_file.token > output_file.ascii
#
# Where VOCAB_LEN is the number of tokens in the vocabulary.
# You must provide VOCAB_LEN as the first argument, and it can be
# 256, 512 or 1024. 
#
# Note that this script selectively loads vocab files from the
# /data/training_data/ directory, depending on the VOCAB_LEN.
# These files were created with "build_vocab.py" by processing
# the full Gutenberg english_corpus.ascii file (18GB).
#   

import sys
from tokenizers import Tokenizer
import numpy as np

CHUNK_SIZE = 1000 # number of tokens to read at a time

# Modify the stdin for reading binary data:
input_file = sys.stdin.buffer
output_file = sys.stdout.buffer

# Check if there are more than 1 argument, if not, exit and print usage:
if len(sys.argv) < 2:
    print("Usage: {} VOCAB_LEN < input_file.token > output_file.ascii \n VOCAB_LEN can be 256, 512 or 1024".format(sys.argv[0]))
    sys.exit(1)

# Grab the first argument as the vocab length:
vocab_len = int(sys.argv[1])

input_data_type = np.uint8 if vocab_len == 256 else np.uint16
vocab_file = "/data/training_data/vocab{}.json".format(vocab_len)

tokenizer = Tokenizer.from_file(vocab_file)
  
# read and detokenize the input file:
# delete_first_char = False
while True:
    # Read a chunk of tokens from stdin:
    tokens = np.frombuffer(input_file.read(CHUNK_SIZE * input_data_type().itemsize), dtype=input_data_type)

    # If tokens is empty, we have reached the end of the input:
    if tokens.size == 0:
        break

    # Detokenize the tokens:
    text = tokenizer.decode(tokens)

    # # For some reason, the detokenized data always has a "*" after each "\n".
    # # So we need to be able to detect this and delete it, but since we aren't
    # # able to process whole lines, we have to do it using a state machine:
    # if delete_first_char:
    #     text = text[1:]
    #     delete_first_char = False
    # if text[-1] == '\n':
    #     delete_first_char = True
    # text = text.replace("\n*", "\n")

    # Write the text to stdout as ascii data, not utf-8:
    output_file.write(text.encode('ascii'))


# Flush the output buffer:
output_file.flush()



