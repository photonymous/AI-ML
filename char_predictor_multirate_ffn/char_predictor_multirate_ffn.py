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
# import a learning rate scheduler:
from torch.optim.lr_scheduler import ExponentialLR

# 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97

# Specify all global constants that aren't arguments:
VOCAB_SIZE     = 256
LEARNING_RATE  = 0.001
LR_GAMMA       = 1

# I need a good variable name for the kernel size of the first convolutional layer. I like to think of it
# in DSP terms, and in DSP you'd call it the "number of taps" or the "FIR length" or the "filter length"
# or the "support". But in machine learning you call it the "kernel size". I'm going to call it the
# "FIFO length" because that's what it is. It's the length of the FIFO that the convolutional layer
# uses to compute its output. It's the number of input samples that the convolutional layer uses to
# compute its output. 

# Specify defaults for all arguments as ALL_CAPS globals:
MODE                = "train"
#                      0	1	  2	    3	      4 	5	  6	    7
#                      1234567890123456789012345678901234567890123456789012345678901234567890
SEED_STR            = "snow rhymes with"   
NUM_CHARS           = 500
EMBEDDING_LEN       = 53
SEQ_LEN             = 59
WARMUP              = 16
NUM_EPOCHS          = 40
FIFO_LEN            = 4 # <-- This is the number of embedded characters that the first convolutional layer uses to compute its output. All subsequent stages reuse this value.
CONVNET_HIDDEN_DIMS = [[61,67],[71,73]] # <-- This is a list of lists. Each list is the hidden dimensions for a convnet. The number of convnets is the length of this list.
PRED_HIDDEN_DIMS    = [257]
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
parser.add_argument('--pred_hidden_dims',    type=int, default=PRED_HIDDEN_DIMS, nargs='+', help='The prediction network hidden dimensions (default: %(default)s)')
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

        self.output_dim = output_dim

        # Create the linear layer:
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, input1, input2):
        # input1 is a tensor of shape (batch_size, seq_len, embedding_len)
        # input2 is a tensor of shape (batch_size, seq_len, convnet_hidden_dims[-1][-1])
        # output is a tensor of shape (batch_size, seq_len, output_dim)

        batch_size = input1.shape[0]
        seq_len = input1.shape[1]
        # Iterate over the sequence:
        output = torch.zeros(batch_size, seq_len, self.output_dim)
        for ii in range(seq_len):
            # Concatenate the input1 and input2 tensors:
            input = torch.cat((input1[:,ii,:], input2[:,ii,:]), dim=1)
            # Compute the output:
            output[:,ii,:] = self.linear(input)

        return output
    

# Define the multistage convolutional network class.
# The initial implementation will just be a single stage.
class MultiStageConvNet(nn.Module):
    def __init__(self, embedding_len, fifo_len, convnet_hidden_dims):
        super(MultistageConvNet, self).__init__()

        # Create the convolutional layers.
        # For now, it will just be a single stage:
        self.conv1 = nn.Conv1d(in_channels=embedding_len, 
                               out_channels=convnet_hidden_dims[0][-1], 
                               kernel_size=fifo_len, 
                               padding=fifo_len-1)
        # Create the ReLU activation layers:
        self.relu1 = nn.ReLU()

    def forward(self, input):
        # input is a tensor of shape (batch_size, seq_len, embedding_len)
        # output is a tensor of shape (batch_size, seq_len, convnet_hidden_dims[-1])
        output = self.conv1(input)
        output = self.relu1(output)
        
        return output


# Define the CharPredictorMultirateFFN class:
class CharPredictorMultirateFFN(nn.Module):
    def __init__(self, vocab_size, embedding_len, seq_len, fifo_len, convnet_hidden_dims, pred_hidden_dims):
        super(CharPredictorMultirateFFN, self).__init__()

        # Create the embedding layer:
        self.embedding = nn.Embedding(vocab_size, embedding_len)

        # Create the multistage convolutional network:
        self.multistage_convnet = MultiStageConvNet(embedding_len, fifo_len, convnet_hidden_dims)

        # Create the prediction network:
        self.pred_net = PredNet(convnet_hidden_dims[-1], pred_hidden_dims, vocab_size)

        # Create the softmax layer:
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, input_sequence):
        # input_sequence is a tensor of shape (batch_size, seq_len)

        # The output of the embedding layer is a tensor of shape (batch_size, seq_len, embedding_len)
        embedding_out = self.embedding(input_sequence)

        # The output of the multistage convolutional network is a tensor of shape (batch_size, seq_len, convnet_hidden_dims[-1])
        convnet_out = self.multistage_convnet(embedding_out)

        # The output of the prediction network is a tensor of shape (batch_size, seq_len, vocab_size)
        prednet_out = self.pred_net(embedding_out, convnet_out)

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
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        model.to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
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
                outputs = model(input_sequences)

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
            
            print(f"Epoch {epoch + 1}/{num_epochs} Loss:{avg_loss:.5f} LR:{old_lr:.5f}->{new_lr:.5f} ETA:{end_time.strftime('%H:%M:%S')} ", flush=True)

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
    model.eval()
    model.to(device)
    context = [ord(c) for c in seed_str]
    
    # Feed one character at a time to the model: (warmup)
    output = None
    for ii in range(len(context)):
        input_tensor = torch.tensor(context[ii], dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)

        # Run the model, but ignore the output for now
        output = model(input_tensor)
    
    # Now we will feed the final output of the model back into the model, one character at a time:
    generated_text = seed_str
    for ii in range(num_chars):
        # Sample a character from the output distribution:
        probs = torch.exp(output).cpu().data.numpy().flatten()
        char_idx = np.random.choice(vocab_size, p=probs)
        generated_text += chr(char_idx)

        # Feed the character back into the model:
        input_tensor = torch.tensor(char_idx, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)
        output = model(input_tensor)

    return generated_text

# This is the main function. It first determines the mode,
# then does what it needs to do based on th emode.
def main():
    if args.mode == "train":
        # Read the corpus:
        corpus = read_corpus(args.corpus_file, args.max_chars)

        # Prepare the input and target sequences:
        input_sequences, target_sequences = prepare_sequences(corpus, args.seq_len)

        # Create a dataloader for the input and target sequences:
        dataloader = create_dataloader(input_sequences, target_sequences, args.batch_size)

        # Create the model:   
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.pred_hidden_dims)


        # Print out the total number of parameters, then the number of weights and biases for each layer, and the
        # number of embedding parameters:
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
        for name, param in model.named_parameters():
            print(f"{name} {param.numel()}", flush=True)
        print(f"Number of embedding parameters: {sum(p.numel() for p in model.embedding_layer.parameters())}", flush=True)
        
        # Train the model:
        train_model(model, dataloader, "cuda" if torch.cuda.is_available() else "cpu", args.num_epochs, args.warmup)

        # Save the model:
        save_model(model, args.model_file)
    elif args.mode == "generate":
        # Create the model:
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.convnet_hidden_dims, args.pred_hidden_dims)

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
    main()




