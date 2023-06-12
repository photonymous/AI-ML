#!/usr/bin/env python3

# This file can be used as an executable at the command line to convert an
# input text file (on stdin, or as the first arguemnt) to a tokenized output.
# It assumes the input is raw 8-bit text, and the ouptut is uint16_t. Each
# output uint16_t maps to a token in the standard GPT-2 50257 token vocabulary.
# Its ouptut is either on stdout or in the file specified by the second argument,
# if a second argument is specified.

import sys
import os
import argparse
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import tiktoken

################################################
def detokenize(tokens, gpt2_encoder=None):
    
    if gpt2_encoder is None:
        gpt2_encoder = tiktoken.encoding_for_model("gpt2")

    # Detect whether the input is a regular list or a numpy array:
    if isinstance(tokens, np.ndarray):
        # Convert the numpy array to a list:
        tokens = tokens.tolist()

    return gpt2_encoder.decode(tokens)

################################################
def tokenize(text, gpt2_encoder=None):

    if gpt2_encoder is None:
        gpt2_encoder = tiktoken.encoding_for_model("gpt2")

    return gpt2_encoder.encode(text, disallowed_special=())

################################################
def main():
    input_file = None
    output_file = None
    # How many arguments were passed in?
    n_args = len(sys.argv)
    if n_args < 2:
        # Check to see that an input is available on stdin:
        if sys.stdin.isatty():
            print("No input file specified and no input on stdin")
            print("Usage: {} <input_file> <output_file>".format(sys.argv[0]))
            sys.exit(1)
        else:
            # Open stdin for input, and stdout for output. Ensure the characters
            # in the file are un-interpreted and no encoding happens
            input_file = sys.stdin

            # Change stdin encoding to latin-1:
            input_file = open(sys.stdin.fileno(), mode='r', encoding='latin-1', closefd=False)

            output_file = sys.stdout.buffer
    elif n_args == 2:
        # Open the file name on argument 1 for input, and stdout for output. Ensure the characters
        # in the file are un-interpreted and no encoding happens
        input_file = open(sys.argv[1], "r", encoding="latin-1")
        output_file = sys.stdout.buffer
    elif n_args == 3:
        # Open the file name on argument 1 for input, and the file name on argument 2 for output:
        input_file = open(sys.argv[1], "r", encoding="latin-1")
        output_file = open(sys.argv[2], "wb")
    else:
        print("Too  many arguments")
        print("Usage: {} <input_file> <output_file>".format(sys.argv[0]))
        sys.exit(1)


    # Read the corpus file, which is raw binary 8-bit text, and
    #  so as to not blow our memory, we'll read the file in chunks of 1MB:
    gpt2_encoder = tiktoken.encoding_for_model("gpt2")
    corpus_piece = None
    chunk = 0
    while corpus_piece != "":        
        # print status to stderr:
        print("Reading 1MB chunk #{}".format(chunk), file=sys.stderr, flush=True, end="\r")
        corpus_piece = input_file.read(2**20)

        # Now tokenize into the 50257 vocabulary:        
        tokens = tokenize(corpus_piece, gpt2_encoder=gpt2_encoder)

        # Convert the tokens list to a numpy array of uint16_t and output to file:
        tokens = np.array(tokens, dtype=np.uint16)
        tokens.tofile(output_file)   

        chunk += 1

    # Close the files:
    input_file.close()
    output_file.close()


################################################
# do the if name is main thing:
if __name__ == "__main__":
    main()


    