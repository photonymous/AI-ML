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

# Specify all global constants that aren't arguments:
VOCAB_SIZE     = 50257
LEARNING_RATE  = 0.001
LR_GAMMA       = 1.0
WEIGHT_DECAY   = 0.1
#USE_AMP        = True # Use Automatic Mixed Precision (AMP) for FP16
# TODO: 
# 1. Add AMP
# 2. Utilize multiple GPUs

# ==================================================================================================
CUDA_DEVICE         = 0
MODE                = "train"
SEED_STR            = """Once upon a time, there was """
TEMPERATURE         = 0.4  
EMBEDDING_LEN       = 64
CONTEXT_LEN         = 16 # tokens
NUM_EPOCHS          = 1
SHUFFLE             = True
NUM_LAYERS          = 4
NUM_HEADS           = 4
HIDDEN_DIM          = 4*EMBEDDING_LEN
BATCH_SIZE          = 128
MAX_CHARS           = 2**24 #2**30
CORPUS_FILE         = "/data/training_data/TinyStories-train.tokens"
MODEL_FILE          = "/data/trained_models/xformer.pth"




# Define our model. Initially, it will just be
# a feed forward network with a single hidden layer
# and a softmax output layer. It will take in the
# context_len tokens as its input, and output a
# distribution over the next token in the sequence.
class TokenPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_len, context_len, num_layers, num_heads, hidden_dim):
        super(TokenPredictor, self).__init__()
        self.vocab_size    = vocab_size
        self.embedding_len = embedding_len
        self.context_len   = context_len
        self.num_layers    = num_layers
        self.num_heads     = num_heads
        self.hidden_dim    = hidden_dim
        

        # Create the embedding layer:
        self.embedding = nn.Embedding(vocab_size, embedding_len)

        ## Create the positional encoding layer:
        #self.positional_encoding = PositionalEncoding(embedding_len, context_len)

        # Create some hidden layers (linear, ReLU):
        layers = []
        current_in_dim = embedding_len
        for i in range(num_layers):
            layers.append(nn.Linear(current_in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.PReLU())
            current_in_dim = hidden_dim
        layers.append(nn.Linear(current_in_dim, vocab_size))
        self.hidden_layers = nn.Sequential(*layers)

        # Initialize the weights using He initialization:
        self.init_weights()

    def forward(self, x):
        # x is a tensor of shape (batch_size, context_len)
        
        # First, we need to embed the tokens:
        x = self.embedding(x) # (batch_size, context_len, embedding_len)

        ## Next, we need to add the positional encoding:
        #x = self.positional_encoding(x) 

        x = self.hidden_layers(x) # (batch_size, vocab_size)

        x = x.reshape(-1, self.vocab_size) 

        # Finally, we need to normalize it:
        x = F.log_softmax(x, dim=1) # (batch_size, vocab_size)

        return x

    def init_weights(self):
        # Initialize the weights using He initialization:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight.data, mean=0, std=sqrt(2.0 / (m.weight.size(0) + m.weight.size(1))))
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)


#############################################################################3
class LazyTokenDataset(Dataset):
    def __init__(self, corpus, context_len):
        self.corpus = corpus
        self.context_len = context_len

    def __len__(self):
        return len(self.corpus) - self.context_len

    def __getitem__(self, idx):
        start = idx * (self.context_len + 1)
        sequence = self.corpus[start:start + self.context_len + 1]
        input = sequence[:-1]
        target = sequence[-1]

        return input, target

def create_dataloader(args, corpus, device):
    # Create the dataset:
    dataset = LazyTokenDataset(corpus, args.context_len)

    # Create the dataloader:
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)

    return dataloader


######################################################################################
class StatusReporter:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.epoch = 0
        self.start_time = time.time()
        self.epoch_start_time = None

    def start_epoch(self, num_batches):
        self.num_batches = num_batches
        self.epoch_start_time = time.time()
        self.epoch += 1
        self.run_avg_loss = 0.0

    def update(self, loss, batch_idx):
        epoch_loss              += loss
        leakage                  = 1.0/(batch_idx+1) if batch_idx < 98 else 0.01
        self.run_avg_loss        = loss*leakage +  self.run_avg_loss*(1.0 - leakage)

        epoch_time_elapsed       = time.time() - self.epoch_start_time
        progress_pct             = batch_idx / self.num_batches * 100
        epoch_remaining_time     = epoch_time_elapsed / progress_pct * (100 - progress_pct)
        epoch_projected_end_time = datetime.datetime.now() + datetime.timedelta(seconds=epoch_remaining_time)
        print(f"\rEpoch {epoch+1:2d} - Progress: {progress_pct:7.3f}%, Loss: {self.run_avg_loss:.5f}, ETA: {epoch_projected_end_time.strftime('%H:%M:%S')}", end="", flush=True)
        
    def finish_epoch():
        print("\r", end="", flush=True)        
        epoch_stop_time    = time.time()
        epoch_elapsed_time = epoch_stop_time - epoch_start_time
        remaining_epochs   = num_epochs - epoch - 1
        remaining_time     = datetime.timedelta(seconds=remaining_epochs * epoch_elapsed_time)
        end_time           = datetime.datetime.now() + remaining_time
        avg_loss           = epoch_loss / num_batches_per_epoch      
        print(f"Epoch {epoch + 1}/{num_epochs} Loss:{run_avg_loss:.4f},{avg_loss:.4f}  dT:{epoch_elapsed_time:6.2f} Finish:{end_time.strftime('%H:%M:%S')} ", flush=True)

    def finish(self):
        stop_time    = time.time()
        elapsed_time = stop_time - start_time
        print(f"Training time: {elapsed_time:.2f} seconds")


###########################################################################
def train(args, device):

    # Read the tokens, which are stored as uint16_t:
    with open(args.corpus_file, "rb") as f:
        corpus = f.read(args.max_chars)
        corpus = np.frombuffer(corpus, dtype=np.uint16)
        corpus = torch.from_numpy(corpus).to(device)

    # Create the model:
    model = TokenPredictor(VOCAB_SIZE, args.embedding_len, args.context_len, args.num_layers, args.num_heads, args.hidden_dim)

    # Print the number of parameters:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    
    # Set the model mode and send it to the device:
    model.train()
    model.to(device)

    # Create the optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Set the loss function:
    criterion = nn.NLLLoss()

    # Create the status reporter:
    status_reporter = StatusReporter(args.num_epochs)

    # Actually do the training:
    for epoch in range(args.num_epochs):

        # Create the data loader:
        data_loader = create_dataloader(args, corpus, device) 


        status_reporter.start_epoch(len(data_loader))

        # Loop over the batches:
        for batch_idx, (input, target) in enumerate(data_loader):
            # Get the batch of data:
            input  = input.to(device)
            target = target.to(device)

            # Zero the gradients:
            optimizer.zero_grad()

            # Forward pass:
            output = model(input)

            # Compute the loss:
            loss = criterion(output, target)

            # Backward pass:
            loss.backward()

            # Update the weights:
            optimizer.step()

            # Update the status reporter:
            status_reporter.update(loss.item(), batch_idx)
        
        status_reporter.finish_epoch()

    status_reporter.finish()

    # Save the model and the optimizer
    torch.save(model.state_dict(), args.model_file)
    torch.save(optimizer.state_dict(), args.model_file + ".opt")



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
        train(args, device)
    elif args.mode == "finetune":
        finetune(args, device)
    elif args.mode == "generate":
        generate(args, device)


# Call the main function:
if __name__ == "__main__":
    main()