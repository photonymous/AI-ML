# This will be a character predicting neural network. From an external vantage point
# it will behave like the CharPredictorRNN (one char in, one char out, with internal 
# state). Internally, however, it will be structured as follows. There will be 
# multiple stages. The first stage is a multilayer feedforward neural network that
# conlvolves along the input data. Its ouptut is updated ever other input sample.
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

# Import the libraries we will need:
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import datetime
from torch.optim.lr_scheduler import ExponentialLR
#import torch.jit
from typing import List
from torch import Tensor
import cProfile
import pstats


# 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97

# Specify all global constants that aren't arguments:
VOCAB_SIZE     = 256
LEARNING_RATE  = 0.001
LR_GAMMA       = 1
WEIGHT_DECAY   = 0.01

# I need a good variable name for the kernel size of the first convolutional layer. I like to think of it
# in DSP terms, and in DSP you'd call it the "number of taps" or the "FIR length" or the "filter length"
# or the "support". But in machine learning you call it the "kernel size". I'm going to call it the
# "FIFO length" because that's what it is. It's the length of the FIFO that the convolutional layer
# uses to compute its output. It's the number of input samples that the convolutional layer uses to
# compute its output. 

# Specify defaults for all arguments as ALL_CAPS globals:
MODE                = "generate"
#                      0	1	  2	    3	      4 	5	  6	    7
#                      1234567890123456789012345678901234567890123456789012345678901234567890
SEED_STR            = "there was a young lady named bright who"
NUM_CHARS           = 500
EMBEDDING_LEN       = 53
SEQ_LEN             = 59 # TODO: This should eventually be a power of 2. And we'll need a CHUNK_LEN that is a power of 2 as well.
WARMUP              = 16
NUM_EPOCHS          = 40
FIFO_LEN            = 4 # <-- This is the number of embedded characters that the first convolutional layer uses to compute its output. All subsequent stages reuse this value.
CONVNET_HIDDEN_DIMS = [[61,67],[71,73]] # <-- This is a list of lists. Each list is the hidden dimensions for a convnet. The number of convnets is the length of this list.
PREDNET_HIDDEN_DIMS = [257]
BATCH_SIZE          = 79
MAX_CHARS           = 2**20
CORPUS_FILE         = "/data/training_data/gutenberg_corpus_21MB.txt"
MODEL_FILE          = "/home/mrbuehler/pcloud/GIT/AI-ML/trained_mrffn.pth"

# Define the command line arguments and assign defaults and format the strings using the globals:
# Note that the arguments can be accessed in code like this: args.mode, args.seed_str, etc.
parser = argparse.ArgumentParser(description='Train or generate text using a character predicting RNN.')
parser.add_argument('--mode',                type=str, default=MODE, help='The mode: train or generate (default: %(default)s)')
parser.add_argument('--seed_str',            type=str, default=SEED_STR, help='The seed string to use for generating text (default: %(default)s)')
parser.add_argument('--num_chars',           type=int, default=NUM_CHARS, help='The number of characters to generate (default: %(default)s)')
parser.add_argument('--embedding_len',       type=int, default=EMBEDDING_LEN, help='The embedding length (default: %(default)s)')
parser.add_argument('--seq_len',             type=int, default=SEQ_LEN, help='The sequence length (default: %(default)s)')
parser.add_argument('--warmup',              type=int, default=WARMUP, help='The warmup (default: %(default)s)')
parser.add_argument('--fifo_len',            type=int, default=FIFO_LEN, help='The FIFO length (default: %(default)s)')
parser.add_argument('--convnet_hidden_dims', type=int, default=CONVNET_HIDDEN_DIMS, nargs='+', help='The convnet hidden dimensions (default: %(default)s)')
parser.add_argument('--prednet_hidden_dims', type=int, default=PREDNET_HIDDEN_DIMS, nargs='+', help='The prediction network hidden dimensions (default: %(default)s)')
parser.add_argument('--num_epochs',          type=int, default=NUM_EPOCHS, help='The number of epochs (default: %(default)s)')
parser.add_argument('--batch_size',          type=int, default=BATCH_SIZE, help='The batch size (default: %(default)s)')
parser.add_argument('--max_chars',           type=int, default=MAX_CHARS, help='The maximum number of characters to read from the corpus file (default: %(default)s)')
parser.add_argument('--corpus_file',         type=str, default=CORPUS_FILE, help='The corpus file (default: %(default)s)')
parser.add_argument('--model_file',          type=str, default=MODEL_FILE, help='The model file (default: %(default)s)')
args = parser.parse_args()


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
            layers.append(nn.ReLU())
            current_in_dim = self.hidden_dims[i]
        layers.append(nn.Linear(current_in_dim, self.output_dim))

        # layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        # for i in range(1, len(self.hidden_dims)):
        #     layers.append(nn.ReLU())
        #     layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))

        self.layers = nn.Sequential(*layers)



      
    def forward(self, input1, input2):
        # input1 is a tensor of shape (batch_size, embedding_len, seq_len)
        #     and is the output of the embedding layer
        # input2 is list of tensors. Each tensor, j, has the shapee (batch_size, convnet_hidden_dims[j][-1], seq_len + padding)
        #    They are the outputs of the last convolutional layers in each convnet.
        # output is a tensor of shape (batch_size, seq_len, output_dim)

        batch_size = input1.shape[0]
        seq_len = input1.shape[2]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        #===============================
        # EXPERIMENTAL: WON'T WORK WHEN WE SCALE UP THE ALGORITHM TO USE MORE THAN ONE CONVNET AND DECIMATION.
        # Instead of a nested for-loop, just use tensor operations to create "input".
        # Concatenate input1 and the tensors in input2 along the 2nd dimension, but only from 0 to seq_len:
        # TODO: Once we add decimation, we'll need a for loop to build up the input tensor.
        #       We will use repeat_interleave() to upsample the decimated convnet outputs to the same length.
        #       Once the context length is too long, we will need to chunk it up and iterate over chunks of
        #       sequence elements, so this will need a second for loop.
        # Construct a tuple containing input1[:,:,:seq_len] and all the tensors in input2 indexed at [:,:,:seq_len]:
        tuple_of_tensors = (input1[:,:,:seq_len],) + tuple(input2[jj][:,:,:seq_len] for jj in range(len(input2)))
        # Concatenate the tensors in the tuple along the 2nd dimension:
        input = torch.cat(tuple_of_tensors, dim=1)
        # Transpose the result to make the shape (batch_size, seq_len, input_dim):
        input = input.transpose(1,2)

        output = self.layers(input)

        return output


# It would be more elegant if we had a ConvNetStage class, and a MultiStageConvNet class that
# contained a list of ConvNetStage objects. Lets define the ConvNetStage class first.
class ConvNetStage(nn.Module):
    def __init__(self, input_dim, fifo_len, hidden_dims):
        super(ConvNetStage, self).__init__()
        
        # define the members:
        self.input_dim   = input_dim
        self.fifo_len    = fifo_len
        self.hidden_dims = hidden_dims

        # define the layers:
        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        # Create the first layer:
        self.conv_layers.append(nn.Conv1d(in_channels=input_dim, out_channels=hidden_dims[0], kernel_size=fifo_len, padding=fifo_len-1))
        self.relu_layers.append(nn.ReLU())
        # Create the rest of the layers:
        for ii in range(1, len(hidden_dims)):
            self.conv_layers.append(nn.Conv1d(in_channels=hidden_dims[ii-1], out_channels=hidden_dims[ii], kernel_size=1))
            self.relu_layers.append(nn.ReLU())

    def forward(self, input):
        # input is a tensor of shape (batch_size, input_dim, seq_len)
        # output is a tensor of shape (batch_size, hidden_dims[-1], seq_len + padding)
        output = input
        for ii in range(len(self.conv_layers)):
            output = self.conv_layers[ii](output)
            output = self.relu_layers[ii](output)
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
        
        # TODO: Add decimation.        
        
        # Create all the stages. We will use a list to hold the stages:
        self.convnet_hidden_dims = convnet_hidden_dims
        self.stages = nn.ModuleList()
        for ii in range(len(convnet_hidden_dims)):
            # The input dimension to the first stage is the embedding length:
            if ii == 0:
                input_dim = embedding_len
            else:
                input_dim = convnet_hidden_dims[ii-1][-1]
            # Create the stage:
            self.stages.append(ConvNetStage(input_dim, fifo_len, convnet_hidden_dims[ii]))
        
        
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

# Prepare the input and target sequences. Remember,
# the input data is unsigned 8-bit values. Every character in the
# input sequence will have a corresponding target character.
def prepare_sequences(corpus, seq_len):
    # Now we need to create the input sequences. Each input sequence is just a little piece of the corpus.
    # If our sequence length is 5, then we actually need to grab 6 characters from the corpus, because
    # the first 5 characters will be the input sequence, and then the 2nd through the 6th characters will
    # be the target sequence. We will use the first 5 characters to predict the 2nd through the 6th characters.
    # So you'll see "seq_len + 1" in the code below prior to pruning down to seq_len.

    corpus_tensor = torch.tensor(list(corpus), dtype=torch.long)
    
    # First, we need to figure out how many batches we can create:
    num_batches = len(corpus_tensor) // (seq_len+1)

    # Now we need to figure out how many characters we need to drop from the end of the corpus tensor
    # so that we can create an input sequence of length seq_len:
    num_chars_to_drop = len(corpus_tensor) - (num_batches * (seq_len+1))

    # Now we need to drop those characters from the end of the corpus tensor:
    if num_chars_to_drop > 0:
        corpus_tensor = corpus_tensor[:-num_chars_to_drop]

    # Now we need to reshape the corpus tensor into a tensor of size (num_batches, seq_len),
    # and we'll use view() so that we don't have to copy the data:
    corpus_tensor = corpus_tensor.view(num_batches, seq_len+1)

    # Finally, form the input and target sequences, and make them seq_len long:
    input_data  = corpus_tensor[:, :-1]
    target_data = corpus_tensor[:, 1:]

    return input_data, target_data

# Create the dataset:
class CorpusDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences  = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]

# Create the dataloader. How does the batch_size relate to the seq_len?
#  The batch_size is the number of input sequences that will be fed to the model per batch.
def create_dataloader(input_sequences, target_sequences, batch_size):
    dataset    = CorpusDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Train the model:
def train_model(model, dataloader, device, num_epochs, warmup):
    #with torch.autograd.set_detect_anomaly(True):
    model.train()
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        for batch_idx, (input_sequences, target_sequences) in enumerate(dataloader):
            # Send the input and target sequences to the device:
            input_sequences  = input_sequences.to(device)
            target_sequences = target_sequences.to(device)

            # Zero the parameter gradients:
            optimizer.zero_grad()

            # Forward pass:
            #with torch.profiler.profile(record_shapes=True) as prof:
            outputs = model(input_sequences)
            #print(prof.key_averages().table(sort_by="cpu_time_total"))

            # Compute the loss:
            outputs = outputs[:, warmup:, :]
            target_sequences = target_sequences[:, warmup:]

            # Now reshape the outputs and targets:            
            outputs = outputs.reshape(-1, VOCAB_SIZE)
            target_sequences = target_sequences.reshape(-1)

            loss = criterion(outputs, target_sequences)

            # Backward pass:
            loss.backward()

            # Update the parameters:
            optimizer.step()

            epoch_loss += loss.item()

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        remaining_epochs = num_epochs - epoch - 1
        remaining_time = datetime.timedelta(seconds=remaining_epochs * elapsed_time)
        end_time = datetime.datetime.now() + remaining_time
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"Epoch {epoch + 1}/{num_epochs} Loss:{avg_loss:.5f} LR:{old_lr:.5f}->{new_lr:.5f} dT:{elapsed_time:6.2f} ETA:{end_time.strftime('%H:%M:%S')} ", flush=True)


# Save the model:
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# Load the model:
def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))

# Generate text using the model. The seed_str is the initial context.
#   Note that the length of the seed_str acts as the "warmup". The model will first
#   be fed with the seed_str, one character at a time, as warmup. Then the model
#   will be fed with its own output, one character at a time to generate the text.
def generate_text(model, seed_str, num_chars, device, vocab_size):
    
    # TODO: Make this more efficient. Right now, it just adds the most recent predicted
    #  character to the generated text, and then feeds the whole generated text to the 
    #  model to get the next character.
    #  So its input gets longer and longer as it runs, and it can't reuse its previous
    #  computations.

    model.eval()
    model.to(device)
    context = [ord(c) for c in seed_str]
    
    # First, we need to feed the model with the seed_str to get the character that it predicts
    # following seed_str. It is a convolutional model, so we can just feed it the whole seed_str.
    context_tensor = torch.tensor(context, dtype=torch.long, device=device)
    context_tensor = context_tensor.unsqueeze(0)
    outputs = model(context_tensor)
    
    # Outputs is the next-character prediction for each character in the seed_str.
    # We need to get the last prediction, which is the prediction for the last character
    # in the seed_str:
    predicted_char = outputs[0, -1, :].argmax().item()
    context.append(predicted_char)

    # Now we can start generating the text:
    generated_text = seed_str + chr(predicted_char)
    for i in range(num_chars):
        # We need to feed the model with the new context to get the next
        # character prediction:
        context_tensor = torch.tensor(context, dtype=torch.long, device=device)
        context_tensor = context_tensor.unsqueeze(0)
        outputs = model(context_tensor)

        # Generate the next character prediction by sampling from the output distribution.
        # We should be using np.random.choice for this, and the outputs have already been softmaxed:
        probs = torch.exp(outputs[0,-1,:]).cpu().data.numpy().flatten()
        predicted_char = np.random.choice(vocab_size, p=probs)
        
        context.append(predicted_char)

        print(i, flush=True)

        # Append the predicted character to the generated text:
        generated_text += chr(predicted_char)

    return generated_text

# This is the main function. It first determines the mode,
# then does what it needs to do based on the emode.
def main():
    if args.mode == "train":
        # Read the corpus:
        corpus = read_corpus(args.corpus_file, args.max_chars)

        # Prepare the input and target sequences:
        input_sequences, target_sequences = prepare_sequences(corpus, args.seq_len)

        # Create a dataloader for the input and target sequences:
        dataloader = create_dataloader(input_sequences, target_sequences, args.batch_size)

        # Create the model:   
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.prednet_hidden_dims)

        # Print out the total number of parameters, then the number of weights and biases for each layer, and the
        # number of embedding parameters:
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
        for name, param in model.named_parameters():
            print(f"{name} {param.numel()}", flush=True)
        
        # Train the model:
        train_model(model, dataloader, "cuda" if torch.cuda.is_available() else "cpu", args.num_epochs, args.warmup)

        # Save the model:
        save_model(model, args.model_file)
    elif args.mode == "generate":
        # Create the model:
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.prednet_hidden_dims)

        # Load the model:
        load_model(model, args.model_file, "cuda" if torch.cuda.is_available() else "cpu")

        # Generate text:
        generated_text = generate_text(model, args.seed_str, args.num_chars, "cuda" if torch.cuda.is_available() else "cpu", VOCAB_SIZE)

        # Print the generated text:
        print(generated_text)
    else:
        print("Invalid mode: " + args.mode)

# Call the main function:
if __name__ == "__main__":
    #with cProfile.Profile() as pr:
    main()

    #pr.print_stats(sort="time")






