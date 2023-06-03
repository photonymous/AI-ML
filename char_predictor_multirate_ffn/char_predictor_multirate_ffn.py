# This will be a character predicting neural network. From an external vantage point
# it will behave like the CharPredictorRNN (one char in, one char out, with internal 
# state). Internally, however, it will be structured as follows. There will be 
# multiple stages. The first stage is a multilayer feedforward neural network that
# conlvolves along the input data. Its ouptut is updated every other input sample.
# The second stage behaves the same way, taking the output of the first stage as
# input and updating every other sample, so its update rate is 1/4th of the initial
# input rate. The third stage behaves the same way, taking the output of the second
# stage as input and updating every other sample, so its update rate is 1/8th of the
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
#import os

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Specify all global constants that aren't arguments:
VOCAB_SIZE     = 256
LEARNING_RATE  = 0.001
LR_GAMMA       = 0.9
WEIGHT_DECAY   = 0.0
USE_AMP        = True # Use Automatic Mixed Precision (AMP) for FP16


# # ==================================================================================================
# # Version 1.0 of a medium-sized (13M parameters) trained model. Produces loss of 0.49.
# # Exhibits good gramatical structure, but doesn't seem to know much about the meaning of words, and
# # doesn't seem to be using the context very well. The subject changes continuously.
# CUDA_DEVICE         = 0
# MODE                = "generate"
# #                         0        1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         
# #                         1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
# SEED_STR            = """John got a new blue ball. He showed it to all his neighbors. He showed it to Lucy, who really liked it. Then he showed it to Ben. Ben did not like the ball becuase he doesn't like the color. They got into an argument about the colo"""
# EMBEDDING_LEN       = 64
# SEQ_LEN             = 2048 
# NUM_EPOCHS          = 10
# SHUFFLE             = True
# FIFO_LEN            = 4 # <-- This is the number of embedded characters that the first convolutional layer uses to compute its output. All subsequent stages reuse this value.
# CONVNET_HIDDEN_DIMS = [[512,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256]] # <-- This is a list of lists. Each list is the hidden dimensions for a convenet stage. The number of convnets in a stage is the length of each list.
# PREDNET_HIDDEN_DIMS = [2048,1024,1024,512,256]
# BATCH_SIZE          = 64
# MAX_CHARS           = 1959000000 #2**30
# #CORPUS_FILE         = "/data/training_data/wiki.train.raw" #"/data/training_data/gutenberg_corpus_21MB.txt"
# CORPUS_FILE         = "/data/training_data/TinyStories-train.txt"
# MODEL_FILE          = "/home/mrbuehler/GIT/AI-ML/trained_models/trained_mrffn_v1.pth"

# # ==================================================================================================
# # Version 2. New "big" model with 26M parameters. Loss is < 0.42.
# CUDA_DEVICE         = 0
# MODE                = "train"
# #                         0        1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         
# #                         1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
# #SEED_STR            = """Once upon a time, there was a little dog named Charlie."""
# SEED_STR            = """Story: Timmy loves trains. One day, he and his dad went to the train station. While at the train station, they read books, explored, and ate lunch. After a long time,"
# """
# TEMPERATURE         = 0.25  # only used for generation. Very low values are very deterministic. 1.0 draws from the exact distribution. Higher values are more random.
# EMBEDDING_LEN       = 64
# SEQ_LEN             = 4096 
# NUM_EPOCHS          = 5
# SHUFFLE             = True
# FIFO_LEN            = 4 # <-- This is the number of embedded characters that the first convolutional layer uses to compute its output. All subsequent stages reuse this value.
# CONVNET_HIDDEN_DIMS = [[512,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256],[256,256,256]] # <-- This is a list of lists. Each list is the hidden dimensions for a convenet stage. The number of convnets in a stage is the length of each list.
# PREDNET_HIDDEN_DIMS = [4096,2048,1024,512,256]
# BATCH_SIZE          = 16
# MAX_CHARS           = 1959000000 #2**26 #2**30
# CORPUS_FILE         = "/data/training_data/TinyStories-train.txt"
# MODEL_FILE          = "/home/mrbuehler/GIT/AI-ML/trained_models/trained_mrffn_v2_7stages.pth"

# ==================================================================================================
# Version 3. Experimenting with AMP (for FP16) 
CUDA_DEVICE         = 0
MODE                = "train"
#                         0        1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         0         1         2         3         4         5         6         7         8         9         
#                         1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
SEED_STR            = """John got"""
TEMPERATURE         = 0.25  # only used for generation. Very low values are very deterministic. 1.0 draws from the exact distribution. Higher values are more random.
EMBEDDING_LEN       = 32
SEQ_LEN             = 256 
NUM_EPOCHS          = 1
SHUFFLE             = True
FIFO_LEN            = 4 # <-- This is the number of embedded characters that the first convolutional layer uses to compute its output. All subsequent stages reuse this value.
CONVNET_HIDDEN_DIMS = [[256,128],[128,128],[128,128],[128,128],[128,128],[128,128]] # <-- This is a list of lists. Each list is the hidden dimensions for a convenet stage. The number of convnets in a stage is the length of each list.
PREDNET_HIDDEN_DIMS = [2048,1024,512,256]
BATCH_SIZE          = 128
MAX_CHARS           = 2**23 #2**26 #2**30
CORPUS_FILE         = "/data/training_data/TinyStories-train.txt"
MODEL_FILE          = "/home/mrbuehler/GIT/AI-ML/trained_models/trained_mrffn_v3.pth"

# Define the command line arguments and assign defaults and format the strings using the globals:
# Note that the arguments can be accessed in code like this: args.mode, args.seed_str, etc.
parser = argparse.ArgumentParser(description='Train or generate text using a character predicting RNN.')
parser.add_argument('--cuda_device',         type=int,   default=CUDA_DEVICE, help='The GPU to run on (0-based) (default: %(default)s)')
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

torch.cuda.set_device(args.cuda_device)

# Define the prediction network class. It will iterate over the sequence, and for each character in the sequence,
# it will predict the next character in the sequence. It will use the output of the last convolutional layer
# and the output of the embedding layer as input. It will just use one linear layer for now.
# Also, by pre-declaring the output tensor and iterating over the sequence length, we will automatically
# be pruning off the last outputs from the convolution, which were invalid for prediction. Our output will
# have a 1:1 mapping between the input and target sequences.
class PredNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(PredNet, self).__init__()

        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.hidden_dims = hidden_dims

        # Create the layers using a list and Sequential(). We need ReLU() between the linear layers, but the
        # last layer should not have a ReLU() after it because we want the output to be a probability distribution
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
        # convenet_hidden_dims description:
        #    [[61,67]],[71,73]] means there are two stages. Each stage has two layers.
        #    The first layer in the first stage has 61 output features. 
        #    The second layer in the last stage has 73 output features. Etc.
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


# Preprocess the corpus data as raw 8-bit binary data:
def read_corpus(file_path, max_chars=None):
    with open(file_path, "rb") as f:
        corpus = f.read(max_chars)
    return corpus


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



def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



# Train the model:
def train_model(model, optimizer, corpus, batch_size, seq_len, device, num_epochs, shuffle):
    # with torch.autograd.set_detect_anomaly(True):    
    criterion = nn.NLLLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_time = time.time()

    # this must take into account the seq_len and the batch_size
    total_num_batches = len(corpus) // (seq_len*batch_size)

    scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        num_batches_completed = 0

        # Because each stage decimates, the top stage ends up seeing a small number of identical examples
        # epoch after epoch. But we can use a data augmentation strategy to mitigate this. Since the corupus is
        # reshaped to create the input sequences, we can start the process at a different character for each
        # epoch. This way each epoch's training set has a different "data phase". But this means we need to 
        # create a new dataloader for each epoch. We can do this by creating a function that returns a dataloader
        # for a given epoch index, where the epoch index is used to determine the starting character for the
        # input sequences:
        dataloader = create_phased_dataloader(epoch, corpus, batch_size, seq_len, device, shuffle) 

        epoch_loss = 0.0
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

            epoch_loss += loss.item()

            num_batches_completed += 1
            progress_pct = num_batches_completed / total_num_batches * 100
            print(f"\rEpoch Progress: {progress_pct:7.3f}%", end="", flush=True)
        print("\r", end="", flush=True)


        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]
        epoch_stop_time = time.time()
        epoch_elapsed_time = epoch_stop_time - epoch_start_time
        remaining_epochs = num_epochs - epoch - 1
        remaining_time = datetime.timedelta(seconds=remaining_epochs * epoch_elapsed_time)
        end_time = datetime.datetime.now() + remaining_time
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"Epoch {epoch + 1}/{num_epochs} Loss:{avg_loss:.5f} LR:{old_lr:.5f}->{new_lr:.5f} dT:{epoch_elapsed_time:6.2f} ETA:{end_time.strftime('%H:%M:%S')} ", flush=True)

    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

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

# Generate text using the model. The seed_str is the initial context.
#   Note that the length of the seed_str acts as the "warmup". The model will first
#   be fed with the seed_str, one character at a time, as warmup. Then the model
#   will be fed with its own output, one character at a time to generate the text.
def generate_text(model, seed_str, temperature, seq_len, device, vocab_size):
    model.eval()
    model.to(device)

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

    print(SEED_STR, ":", end="", flush=True)
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


# This is the main function. It first determines the mode,
# then does what it needs to do based on the emode.
def main():
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
        
        # Train the model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.train()
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-4) # eps is used to prevent NaNs in the loss
        train_model(model, optimizer, corpus, args.batch_size, args.seq_len, device, args.num_epochs, args.shuffle)

        # Save the model and optimizer:
        save_model(model, args.model_file)
        save_optimizer(optimizer, args.model_file + ".opt")

    elif args.mode == "finetune":
        # Read the corpus:
        corpus = read_corpus(args.corpus_file, args.max_chars)

        # Create the model:
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.prednet_hidden_dims)

        # Load the model:
        load_model(model, args.model_file, "cuda" if torch.cuda.is_available() else "cpu")

        # Train the model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.train()
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-4) # eps is used to prevent NaNs in the loss
        load_optimizer(optimizer, args.model_file + ".opt", device)
        train_model(model, optimizer, corpus, args.batch_size, args.seq_len, device, args.num_epochs, args.shuffle)

        # Save the model and optimizer:
        save_model(model, args.model_file)
        save_optimizer(optimizer, args.model_file + ".opt")

    elif args.mode == "generate":
        # Create the model:
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.prednet_hidden_dims)

        # Load the model:
        load_model(model, args.model_file, "cuda" if torch.cuda.is_available() else "cpu")

        # Generate text:
        generated_text = generate_text(model, args.seed_str, args.temperature, args.seq_len, "cuda" if torch.cuda.is_available() else "cpu", VOCAB_SIZE)

    else:
        print("Invalid mode: " + args.mode)

# Call the main function:
if __name__ == "__main__":
    main()



