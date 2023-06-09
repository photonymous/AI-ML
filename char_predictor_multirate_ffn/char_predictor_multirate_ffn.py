#!/usr/bin/env python3

# This is a character predicting neural network. There are 
# multiple stages. The first stage is a multilayer feedforward neural network that
# conlvolves along the input data. Its ouptut is updated every input sample.
# The second stage behaves the same way, taking the output of the first stage as
# input but updating every other sample, so its update rate is 1/2th of the initial
# input rate. The third stage behaves the same way, taking the output of the second
# stage as input and updating every other sample, so its update rate is 1/4th of the
# initial input rate. And so on. The final stage is a prediction network that takes
# its input from all of the previous stages' most recent outputs, including the
# embedding layer which provides the initial input to the network. The prediction
# network is a multilayer feedforward neural network that outputs a probability
# distribution over the next character. The prediction network is updated every
# input sample. The prediction network is trained to predict the next character
# in the sequence. The loss function is the cross entropy loss function. 

# Use the following bash command to get seed strings of a specific length. In this 
# case, 512 characters. You can modify the "skip" parameter to jump through the
# file in increments of this many characters:
#       tr -d '\n\r' < wiki.valid.raw | dd bs=512 skip=11 count=1 2>/dev/null | wc

# Training data:
#  Children's book corpus: https://huggingface.co/roneneldan/
#  Standardized Gutenberg corpus: https://github.com/pgcorpus/gutenberg

# Import the libraries we will need:
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


# Specify all global constants that aren't arguments:
VOCAB_SIZE     = 256
LEARNING_RATE  = 0.001
LR_GAMMA       = 1.0
WEIGHT_DECAY   = 0.1
USE_AMP        = True # Use Automatic Mixed Precision (AMP) for FP16

# ==================================================================================================
CUDA_DEVICE         = -1
MODE                = "generate"
SEED_STR            = """read the book read the book read the book"""
TEMPERATURE         = 0.4  
EMBEDDING_LEN       = 32
SEQ_LEN             = 256 
NUM_EPOCHS          = 1
SHUFFLE             = True
FIFO_LEN            = 4 
CONVNET_HIDDEN_DIMS = [[256,128],[128,128],[128,128],[128,128],[128,128],[128,128]] 
PREDNET_HIDDEN_DIMS = [1024,512,256]
BATCH_SIZE          = 128
MAX_CHARS           = 2**24 #2**30
CORPUS_FILE         = "/data/training_data/TinyStories-train.txt"
MODEL_FILE          = "/data/trained_models/small.pth"

# Define the command line arguments and assign defaults and format the strings using the globals:
# Note that the arguments can be accessed in code like this: args.mode, args.seed_str, etc.
parser = argparse.ArgumentParser(description='Train or generate text using a character predicting network.')
parser.add_argument('--cuda_device',         type=int,   default=CUDA_DEVICE, help='The GPU to run on. -1 = use all GPUs. (default: %(default)s)')
parser.add_argument('--mode',                type=str,   default=MODE, help='The mode: train, finetune, or generate (default: %(default)s)')
parser.add_argument('--seed_str',            type=str,   default=SEED_STR, help='The seed string to use for generating text (default: %(default)s)')
parser.add_argument('--temperature',         type=float, default=TEMPERATURE, help='The temperature to use for generating text (default: %(default)s)')
parser.add_argument('--embedding_len',       type=int,   default=EMBEDDING_LEN, help='The embedding length (default: %(default)s)')
parser.add_argument('--seq_len',             type=int,   default=SEQ_LEN, help='The sequence length (default: %(default)s)')
parser.add_argument('--fifo_len',            type=int,   default=FIFO_LEN, help='The FIFO length (default: %(default)s)')
parser.add_argument('--convnet_hidden_dims', type=ast.literal_eval,   default=CONVNET_HIDDEN_DIMS, help='The convnet hidden dimensions (default: %(default)s)')
parser.add_argument('--prednet_hidden_dims', type=ast.literal_eval,   default=PREDNET_HIDDEN_DIMS, help='The prediction network hidden dimensions (default: %(default)s)')
parser.add_argument('--num_epochs',          type=int,   default=NUM_EPOCHS, help='The number of epochs (default: %(default)s)')
parser.add_argument('--shuffle',             type=bool,  default=SHUFFLE, help='Whether to shuffle the data (default: %(default)s)')
parser.add_argument('--batch_size',          type=int,   default=BATCH_SIZE, help='The batch size (default: %(default)s)')
parser.add_argument('--max_chars',           type=int,   default=MAX_CHARS, help='The maximum number of characters to read from the corpus file (default: %(default)s)')
parser.add_argument('--corpus_file',         type=str,   default=CORPUS_FILE, help='The corpus file (default: %(default)s)')
parser.add_argument('--model_file',          type=str,   default=MODEL_FILE, help='The model file (default: %(default)s)')
args = parser.parse_args()

if args.cuda_device > -1:
    torch.cuda.set_device(args.cuda_device)

###########################################################################################################################

# Define the prediction network class. It will iterate over the sequence, and for each character in the sequence,
# it will predict the next character in the sequence. It will use the most recent output of each stage, as well
# as the output of the embedding layer as input. 
class PredNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(PredNet, self).__init__()

        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.hidden_dims = hidden_dims

        # Create the layers using a list and Sequential(). We need PReLU() between the linear layers, but the
        # last layer should not have a PReLU() after it because we want the output to be a probability distribution
        # and softmax will be applied to it later in before the loss function. Also, the last layer should have
        # dimensionality equal to the number of characters in the vocabulary and is its own special layer. If
        # hidden_dims is empty we will still have this output layer.
        layers = []
        current_in_dim = self.input_dim
        for i in range(len(self.hidden_dims)):
            layers.append(nn.Linear(current_in_dim, self.hidden_dims[i]))
            layers.append(nn.LayerNorm(self.hidden_dims[i], eps=1e-4)) # eps is used to prevent NaNs in the loss
            layers.append(nn.PReLU())
            current_in_dim = self.hidden_dims[i]
        layers.append(nn.Linear(current_in_dim, self.output_dim))

        self.layers = nn.Sequential(*layers)
      
    def forward(self, input1, input2):
        # input1 is a tensor of shape (batch_size, embedding_len, seq_len)
        #     and is the output of the embedding layer
        # input2 is list of tensors. Each tensor, j, has the shapee (batch_size, convnet_hidden_dims[j][-1], seq_len + padding)
        #    They are the outputs of the last convolutional layers in each convnet.
        # output is a tensor of shape (batch_size, seq_len, output_dim)

        seq_len = input1.shape[2]

        #tuple_of_tensors = (input1[:,:,:seq_len],) + tuple(input2[jj][:,:,:seq_len] for jj in range(len(input2)))
    
        tuple_of_tensors = (input1[:,:,:seq_len],)
        for jj in range(len(input2)):
            # Upsample the convnet output by the appropriate factor:
#            upsampled = input2[jj].repeat_interleave(2**(jj+1), dim=2)[:,:,:seq_len]
            upsampled = input2[jj].repeat_interleave(2**jj, dim=2)[:,:,:seq_len]
            # Add it to the tuple:
            tuple_of_tensors = tuple_of_tensors + (upsampled,)

        # TODO: Once the context length is too long, we will need to chunk it up and iterate over chunks of
        #       sequence elements, so this will need a second for loop.

        # Concatenate the tensors in the tuple along the 2nd dimension:
        input = torch.cat(tuple_of_tensors, dim=1)
        # Transpose the result to make the shape (batch_size, seq_len, input_dim):
        input = input.transpose(1,2)

        output = self.layers(input)

        return output



class TimeStepNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(TimeStepNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))  # Scale
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))  # Shift
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        return self.gamma * x + self.beta



# It would be more elegant if we had a ConvNetStage class, and a MultiStageConvNet class that
# contained a list of ConvNetStage objects. Lets define the ConvNetStage class first.
class ConvNetStage(nn.Module):
    def __init__(self, input_dim, fifo_len, hidden_dims, dec_by):
        super(ConvNetStage, self).__init__()
        
        # define the members:
        self.input_dim   = input_dim
        self.fifo_len    = fifo_len
        self.hidden_dims = hidden_dims
        self.dec_by      = dec_by

        layers = []

        layers.append(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dims[0], kernel_size=fifo_len, padding=fifo_len-1, stride=dec_by))
        #layers.append(TimeStepNorm(hidden_dims[0]))
        layers.append(nn.PReLU())
        for ii in range(1, len(hidden_dims)):
            layers.append(nn.Conv1d(in_channels=hidden_dims[ii-1], out_channels=hidden_dims[ii], kernel_size=1))
            if ii == len(hidden_dims)-1:
                layers.append(TimeStepNorm(hidden_dims[ii]))
            layers.append(nn.PReLU())
        
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        # input is a tensor of shape (batch_size, input_dim, seq_len)
        # output is a tensor of shape (batch_size, hidden_dims[-1], seq_len//dec_by)

        output = self.layers(input)

        return output



# Define the multistage convolutional network class.
class MultiStageConvNet(nn.Module):
    def __init__(self, embedding_len, fifo_len, convnet_hidden_dims):
        super(MultiStageConvNet, self).__init__()
        # The first layer in each stage will have a kernel size of fifo_len, and
        # the rest of the layers in the stage will have a kernel size of 1.
        
        # Create all the stages. We will use a list to hold the stages:
        self.convnet_hidden_dims = convnet_hidden_dims
        self.stages = nn.ModuleList()
        for ii in range(len(convnet_hidden_dims)):
            # The input dimension to the first stage is the embedding length:
            if ii == 0:
                input_dim = embedding_len
                self.stages.append(ConvNetStage(input_dim, fifo_len, convnet_hidden_dims[ii], dec_by=1))
            else:
                input_dim = convnet_hidden_dims[ii-1][-1]
                self.stages.append(ConvNetStage(input_dim, fifo_len, convnet_hidden_dims[ii], dec_by=2))
        
        
    def forward(self, x):
        # Create a container to hold the stage outputs:
        stage_outputs = []
        # Iterate over the stages:
        for ii in range(len(self.stages)):
            # Get the output from this stage:
            x = self.stages[ii](x)
            # Append the output to the list:
            stage_outputs.append(x)
            
        # Return the list of stage outputs:
        return stage_outputs
    

# Define the CharPredictorMultirateFFN class:
class CharPredictorMultirateFFN(nn.Module):
    def __init__(self, vocab_size, embedding_len, seq_len, fifo_len, convnet_hidden_dims, prednet_hidden_dims):
        super(CharPredictorMultirateFFN, self).__init__()

        # Create the embedding layer:
        self.embedding = nn.Embedding(vocab_size, embedding_len)

        # Create the multistage convolutional network:
        self.multistage_convnet = MultiStageConvNet(embedding_len, fifo_len, convnet_hidden_dims)

        # Create the prediction network:
        #   But first, compute the size of the input dimension to the prediction network, which
        #   is the sum of the embedding length and the size of all of the last convolutional layers:
        prednet_input_dim = embedding_len + sum([convnet_hidden_dims[ii][-1] for ii in range(len(convnet_hidden_dims))])
        self.prednet = PredNet(prednet_input_dim, prednet_hidden_dims, vocab_size)

        # Create the softmax layer: 
        self.softmax = nn.LogSoftmax(dim=2)

    
    def forward(self, input_sequence):
        # input_sequence is a tensor of shape (batch_size, seq_len)
        # The output of the embedding layer is a tensor of shape (batch_size, seq_len, embedding_len)        
        embedding_out = self.embedding(input_sequence)

        # But we want the output to be of shape (batch_size, embedding_len, seq_len) so that it can be fed into the convolutional layer.
        # So, we transpose the tensor: 
        embedding_out = embedding_out.transpose(1,2)

        # The output of the multistage convolutional network is a tensor of shape (batch_size, convnet_hidden_dims[-1], seq_len + padding)
        convnet_out = self.multistage_convnet(embedding_out)

        # The output of the prediction network is a tensor of shape (batch_size, seq_len, vocab_size)
        prednet_out = self.prednet(embedding_out, convnet_out)

        # The output of the softmax layer is a tensor of shape (batch_size, seq_len, vocab_size)
        softmax_out = self.softmax(prednet_out)

        return softmax_out


###########################################################################################################################


# Create the dataset:
class CorpusDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences  = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]


class LazyCorpusDataset(Dataset):
    def __init__(self, corpus, seq_len):
        self.corpus = corpus
        self.seq_len = seq_len

    def __len__(self):
        return len(self.corpus) // (self.seq_len + 1)

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        sample = self.corpus[start : start + self.seq_len + 1]
        input_data = torch.tensor(list(sample[:-1]), dtype=torch.long)
        target_data = torch.tensor(list(sample[1:]), dtype=torch.long)
        return input_data, target_data


def create_phased_dataloader(epoch, corpus, batch_size, seq_len, device, shuffle):
    corpus_at_phase = corpus[epoch:]
    dataset = LazyCorpusDataset(corpus_at_phase, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


###########################################################################################################################

# Initialize the weights using "He" initialization:
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Preprocess the corpus data as raw 8-bit binary data:
def read_corpus(file_path, max_chars=None):
    with open(file_path, "rb") as f:
        corpus = f.read(max_chars)
    return corpus

# Save the model and optimizer:
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def save_optimizer(optimizer, file_path):
    torch.save(optimizer.state_dict(), file_path)

# Load the model and optimizer:
def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))

def load_optimizer(optimizer, file_path, device):
    optimizer.load_state_dict(torch.load(file_path, map_location=device))

###########################################################################################################################
# Train the model:
def train_model(model, optimizer, corpus, batch_size, seq_len, device, num_epochs, shuffle):
    # with torch.autograd.set_detect_anomaly(True):    
    criterion = nn.NLLLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_time = time.time()

    scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        num_batches_completed = 0

        # Because each stage decimates, the top stage ends up seeing a small number of identical examples
        # epoch after epoch. But we can use a data augmentation strategy to mitigate this. Since the corupus is
        # reshaped to create the input sequences, we can start the process at a different character for each
        # epoch. This way each epoch's training set has a different "data phase". 
        # Get a random integer from 0 to seq_len:
        dataset_phase = random.randint(0, seq_len - 1) if shuffle else epoch
        dataloader = create_phased_dataloader(dataset_phase, corpus, batch_size, seq_len, device, shuffle) 

        num_batches_per_epoch = len(dataloader)
        run_avg_loss          = 0.0
        epoch_loss            = 0.0
        for batch_idx, (input_sequences, target_sequences) in enumerate(dataloader):
            # Send the input and target sequences to the device:
            input_sequences  = input_sequences.to(device)
            target_sequences = target_sequences.to(device)

            # Zero the parameter gradients:
            optimizer.zero_grad()

            # Forward pass using AMP for FP16:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=USE_AMP):
                outputs = model(input_sequences)

                # Now reshape the outputs and targets:            
                outputs = outputs.reshape(-1, VOCAB_SIZE)
                target_sequences = target_sequences.reshape(-1)

                loss = criterion(outputs, target_sequences)
            
            # Backward pass:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss               = loss.item()
            epoch_loss            += cur_loss
            num_batches_completed += 1
            leakage                = 1.0/(batch_idx+1) if batch_idx < 98 else 0.01
            run_avg_loss           = cur_loss*leakage +  run_avg_loss*(1.0 - leakage)

            epoch_time_elapsed       = time.time() - epoch_start_time
            progress_pct             = num_batches_completed / num_batches_per_epoch * 100
            epoch_remaining_time     = epoch_time_elapsed / progress_pct * (100 - progress_pct)
            epoch_projected_end_time = datetime.datetime.now() + datetime.timedelta(seconds=epoch_remaining_time)
            print(f"\rEpoch {epoch+1:2d} - Progress: {progress_pct:7.3f}%, Loss: {run_avg_loss:.5f}, ETA: {epoch_projected_end_time.strftime('%H:%M:%S')}", end="", flush=True)
        print("\r", end="", flush=True)


        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]

        epoch_stop_time    = time.time()
        epoch_elapsed_time = epoch_stop_time - epoch_start_time
        remaining_epochs   = num_epochs - epoch - 1
        remaining_time     = datetime.timedelta(seconds=remaining_epochs * epoch_elapsed_time)
        end_time           = datetime.datetime.now() + remaining_time
        avg_loss           = epoch_loss / num_batches_per_epoch
        
        print(f"Epoch {epoch + 1}/{num_epochs} Loss:{run_avg_loss:.4f},{avg_loss:.4f} LR:{old_lr:.5f}->{new_lr:.5f} dT:{epoch_elapsed_time:6.2f} Finish:{end_time.strftime('%H:%M:%S')} ", flush=True)

    stop_time    = time.time()
    elapsed_time = stop_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")


###########################################################################################################################
# Generate text using the model. The seed_str is the initial context.
#   Note that the length of the seed_str acts as the "warmup". The model will first
#   be fed with the seed_str, one character at a time, as warmup. Then the model
#   will be fed with its own output, one character at a time to generate the text.
@torch.no_grad()
def generate_text(model, seed_str, temperature, seq_len, device, vocab_size):    

    # Create a context string filled with spaces that is seq_len long:
    context = [ord(' ')] * seq_len

    # Copy the seed_str into the front of the context string:
    for i, c in enumerate(seed_str):
        context[i] = ord(c)

    context_tensor = torch.tensor(context, dtype=torch.long, device=device)
    context_tensor = context_tensor.unsqueeze(0)
    outputs = model(context_tensor)
    
    # Outputs is the next-character prediction for each character in the context.
    # However, only one of these is valid, the one directly after the last character
    # in the seed_str. So we need to find that one and use it as the first character
    predicted_char_idx = len(seed_str)-1
    predicted_char = outputs[0, predicted_char_idx, :].argmax().item()
    
    # Now we need to replace the corresponding character in the context with the
    # predicted character:
    context[predicted_char_idx+1] = predicted_char
    predicted_char_idx += 1    

    print(seed_str, ":", chr(predicted_char), end="", flush=True)
    # Now we can start generating the text. We'll do this by filling the rest of the context    
    while predicted_char_idx < seq_len-1:
        context_tensor = torch.tensor(context, dtype=torch.long, device=device)
        context_tensor = context_tensor.unsqueeze(0)
        outputs = model(context_tensor) 

        # The output of our model has LogSoftmax() applied already, but we went to use a temperature parameter to
        # scale the logits before we apply softmax. So we need to first exponentiate the outputs, then scale them
        # by the temperature, then apply softmax. We can do this all in one step using the softmax function. Then
        # we can use np.random.choice() to sample from the output distribution.
        probs = F.softmax(outputs[0,predicted_char_idx,:]/temperature, dim=0).cpu().data.numpy().flatten()
        predicted_char = np.random.choice(vocab_size, p=probs)
        
        context[predicted_char_idx+1] = predicted_char
        predicted_char_idx += 1

        # print the progress, overwritng the previous output:
        print(chr(predicted_char), end="", flush=True)
    print("\n")   


###########################################################################################################################
# This is the main function. It first determines the mode,
# then does what it needs to do based on the emode.
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        # Read the corpus:
        corpus = read_corpus(args.corpus_file, args.max_chars)

        # Create the model:   
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.prednet_hidden_dims)

        # Initialize the weights:
        model.apply(init_weights)

        # Print out the total number of parameters, then the number of weights and biases for each layer, and the
        # number of embedding parameters:
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
        for name, param in model.named_parameters():
            print(f"{name} {param.numel()}", flush=True)
              
        if args.cuda_device < 0:
            model = nn.DataParallel(model)
        model.train()
        model.to(device)
        
        # Create the optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-4) # eps is used to prevent NaNs in the loss

        # Train the model:
        train_model(model, optimizer, corpus, args.batch_size, args.seq_len, device, args.num_epochs, args.shuffle)

        # Save the model and optimizer:
        save_model(model, args.model_file)
        save_optimizer(optimizer, args.model_file + ".opt")

    elif args.mode == "finetune":
        # Read the corpus:
        corpus = read_corpus(args.corpus_file, args.max_chars)

        # Create the model:
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.prednet_hidden_dims)
        if args.cuda_device < 0:
            model = nn.DataParallel(model)
        model.train()
        model.to(device)

        # Load the model:
        load_model(model, args.model_file, device)

        # Create the optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-4) # eps is used to prevent NaNs in the loss
        load_optimizer(optimizer, args.model_file + ".opt", device)

        # Train the model:
        train_model(model, optimizer, corpus, args.batch_size, args.seq_len, device, args.num_epochs, args.shuffle)

        # Save the model and optimizer:
        save_model(model, args.model_file)
        save_optimizer(optimizer, args.model_file + ".opt")

    elif args.mode == "generate":
        # Check to see of anything was passed in on stdin:
        read_stdin = False
        if select.select([sys.stdin,],[],[],0.0)[0]:
            read_stdin = True

        # If something was passed in on stdin, then use it to replace the seed_str:
        seed_str = args.seed_str
        if read_stdin:
            seed_str = sys.stdin.read()
        
        # Create the model:
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.prednet_hidden_dims) 
        if args.cuda_device < 0:
            model = nn.DataParallel(model)
        model.eval()
        model.to(device)

        # Load the model:
        load_model(model, args.model_file, device)

        # Generate text:
        generated_text = generate_text(model, seed_str, args.temperature, args.seq_len, device, VOCAB_SIZE)

    else:
        print("Invalid mode: " + args.mode)

# Call the main function:
if __name__ == "__main__":
    main()



