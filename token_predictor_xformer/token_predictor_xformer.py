#!/usr/bin/env python3

# This file implements the Token Predictor Transformer model.
# It uses a basic standard implementation of the transformer found in the
# PyTorch library. It uses the standard GPT-2 tokenization encoding, 
# with a vocabulary length of 50257 and the standard positional encoding
# that uses sines and cosines. It uses the standard PyTorch AdamW optimizer.
# It predicts the distribution of the next token in the sequence, given the
# previous tokens in the sequence. It uses a cross entropy loss function.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import argparse
import ast
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import datetime
from torch.optim.lr_scheduler import ExponentialLR
from typing import List
from torch import Tensor
import cProfile
import pstats
import select
import random
from math import sqrt
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# ==================================================================================================
CUDA_DEVICE         = 0
MODE                = "generate"
SEED_STR            = """Once upon a time, there was """
TEMPERATURE         = 0.4  
EMBEDDING_LEN       = 32
CONTEXT_LEN         = 256 # tokens
NUM_EPOCHS          = 1
SHUFFLE             = True
NUM_LAYERS          = 4
NUM_HEADS           = 4
HIDDEN_DIM          = 4*EMBEDDING_LEN
BATCH_SIZE          = 128
MAX_CHARS           = 2**24 #2**30
CORPUS_FILE         = "/data/training_data/TinyStories-train.txt"
MODEL_FILE          = "/data/trained_models/xformer.pth"


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


def train(model, device):
    # Read the corpus file, which is raw binary 8-bit text:
    with open(args.corpus_file, "rb") as f:
        corpus = f.read(args.max_chars)

        # Now tokenize into the 50257 vocabulary:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = tokenizer.encode(corpus)
        tokens = torch.tensor(tokens).to(device)

        # Create the training data:
        dataset = TensorDataset(tokens)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

def create_model(args, device):
    # Create the model:
    model = TransformerModel(ntoken=50257, ninp=args.ninp, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers, dropout=args.dropout).to(device)

    # Load the model if it exists:
    if os.path.exists(args.model_file):
        print("Loading model from {}".format(args.model_file))
        model.load_state_dict(torch.load(args.model_file))

    return model


###########################################################################################################################
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the command line arguments and assign defaults and format the strings using the globals:
    # Note that the arguments can be accessed in code like this: args.mode, args.seed_str, etc.
    parser = argparse.ArgumentParser(description='Train or generate text using a character predicting network.')
    parser.add_argument('--cuda_device',         type=int,   default=CUDA_DEVICE, help='The GPU to run on. -1 = use all GPUs. (default: %(default)s)')
    parser.add_argument('--mode',                type=str,   default=MODE, help='The mode: train, finetune, or generate (default: %(default)s)')
    parser.add_argument('--seed_str',            type=str,   default=SEED_STR, help='The seed string to use for generating text (default: %(default)s)')
    parser.add_argument('--temperature',         type=float, default=TEMPERATURE, help='The temperature to use for generating text (default: %(default)s)')
    parser.add_argument('--embedding_len',       type=int,   default=EMBEDDING_LEN, help='The embedding length (default: %(default)s)')
    parser.add_argument('--context_len',         type=int,   default=SEQ_LEN, help='The context length (tokens). (default: %(default)s)')
    parser.add_argument('--num_layers',          type=int,   default=NUM_LAYERS, help='The number of layers (default: %(default)s)')
    parser.add_argument('--num_heads',           type=int,   default=NUM_HEADS, help='The number of heads (default: %(default)s)')
    parser.add_argument('--hidden_dim',          type=int,   default=HIDDEN_DIM, help='The hidden dimension (default: %(default)s)')
    parser.add_argument('--num_epochs',          type=int,   default=NUM_EPOCHS, help='The number of epochs (default: %(default)s)')
    parser.add_argument('--shuffle',             type=bool,  default=SHUFFLE, help='Whether to shuffle the data (default: %(default)s)')
    parser.add_argument('--batch_size',          type=int,   default=BATCH_SIZE, help='The batch size (default: %(default)s)')
    parser.add_argument('--max_chars',           type=int,   default=MAX_CHARS, help='The maximum number of characters to read from the corpus file (default: %(default)s)')
    parser.add_argument('--corpus_file',         type=str,   default=CORPUS_FILE, help='The corpus file (default: %(default)s)')
    parser.add_argument('--model_file',          type=str,   default=MODEL_FILE, help='The model file (default: %(default)s)')
    args = parser.parse_args()

if args.cuda_device > -1:
    torch.cuda.set_device(args.cuda_device)

    if args.mode == "train":
        model = create_model(args, device)
        train(model, device)
    elif args.mode == "finetune":
        finetune(device)
    elif args.mode == "generate":
        generate(device)


# Call the main function:
if __name__ == "__main__":
    main()