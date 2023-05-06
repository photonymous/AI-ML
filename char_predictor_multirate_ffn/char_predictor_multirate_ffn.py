# This will be a character predicting neural network. From an external vantage point
# it will behave like the CharPredictorRNN (one char in, one char out, with internal 
# state). Internally, however, it will be a feed forward multirate architecture, using
# FIFOs between subnetworks. The subnetworks are connected in a daisy-chain. Each
# updates at half the rate of the subnetwork before it. As with CharPredictorRNN, 
# it will be trained using a large corpus of
# raw text, and will be able to generate new text based on a seed string. It will process
# one character at a time, and will output a probability distribution over the next
# character. The output will be a softmax over the vocabulary size (256 characters).
# The input will be embedded, with a specifiable embedding length. 

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

# Specify defaults for all arguments as ALL_CAPS globals:
MODE               = "train"
#                     0        1         2         3         4         5         6         7
#                     1234567890123456789012345678901234567890123456789012345678901234567890
SEED_STR           = "snow rhymes with"   
NUM_CHARS          = 500
EMBEDDING_LEN      = 53
SEQ_LEN            = 59
WARMUP             = 16
NUM_EPOCHS         = 40
FIFO_LEN           = 4
SUBNET_HIDDEN_DIMS = [[61,67],[71,73]] # <-- This is a list of lists. Each list is the hidden dimensions for a subnet. The number of subnets is the length of this list.
PRED_HIDDEN_DIMS   = [257]
BATCH_SIZE         = 79
MAX_CHARS          = 2**20
CORPUS_FILE        = "/data/training_data/gutenberg_corpus_21MB.txt"
MODEL_FILE         = "/home/mrbuehler/pcloud/GIT/AI-ML/trained_mrffn.pth"

# Define the command line arguments and assign defaults and format the strings using the globals:
# Note that the arguments can be accessed in code like this: args.mode, args.seed_str, etc.
parser = argparse.ArgumentParser(description='Train or generate text using a character predicting RNN.')
parser.add_argument('--mode',               type=str, default=MODE, help='The mode: train or generate (default: %(default)s)')
parser.add_argument('--seed_str',           type=str, default=SEED_STR, help='The seed string to use for generating text (default: %(default)s)')
parser.add_argument('--num_chars',          type=int, default=NUM_CHARS, help='The number of characters to generate (default: %(default)s)')
parser.add_argument('--embedding_len',      type=int, default=EMBEDDING_LEN, help='The embedding length (default: %(default)s)')
parser.add_argument('--seq_len',            type=int, default=SEQ_LEN, help='The sequence length (default: %(default)s)')
parser.add_argument('--warmup',             type=int, default=WARMUP, help='The warmup (default: %(default)s)')
parser.add_argument('--fifo_len',           type=int, default=FIFO_LEN, help='The FIFO length (default: %(default)s)')
parser.add_argument('--subnet_hidden_dims', type=int, default=SUBNET_HIDDEN_DIMS, nargs='+', help='The subnet hidden dimensions (default: %(default)s)')
parser.add_argument('--pred_hidden_dims',   type=int, default=PRED_HIDDEN_DIMS, nargs='+', help='The prediction network hidden dimensions (default: %(default)s)')
parser.add_argument('--num_epochs',         type=int, default=NUM_EPOCHS, help='The number of epochs (default: %(default)s)')
parser.add_argument('--batch_size',         type=int, default=BATCH_SIZE, help='The batch size (default: %(default)s)')
parser.add_argument('--max_chars',          type=int, default=MAX_CHARS, help='The maximum number of characters to read from the corpus file (default: %(default)s)')
parser.add_argument('--corpus_file',        type=str, default=CORPUS_FILE, help='The corpus file (default: %(default)s)')
parser.add_argument('--model_file',         type=str, default=MODEL_FILE, help='The model file (default: %(default)s)')
args = parser.parse_args()

# Define the subnetwork class:
class Subnetwork(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(Subnetwork, self).__init__()
        
        layers = [nn.Linear(input_size, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()])
        self.ffn = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.ffn(x)

""" # Define the subnetwork class:
class Subnetwork(nn.Module):
    def __init__(self, input_size, hidden_dims, fifo_len):
        super(Subnetwork, self).__init__()
        self.fifo_len = fifo_len
        
        layers = [nn.Linear(input_size * fifo_len, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()])
        self.ffn = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.ffn(x) """

# Define the CharPredictorMultirateFFN class:
class CharPredictorMultirateFFN(nn.Module):
    def __init__(self, vocab_size, embedding_len, seq_len, fifo_len, subnet_hidden_dims, pred_hidden_dims):
        super(CharPredictorMultirateFFN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_len = embedding_len
        self.seq_len = seq_len
        self.num_subnets = len(subnet_hidden_dims)
        self.subnet_hidden_dims = subnet_hidden_dims
        self.fifo_len = fifo_len

        self.embedding_layer = nn.Embedding(vocab_size, embedding_len)

        # Create the subnets:
        subnets = [Subnetwork(embedding_len * fifo_len, subnet_hidden_dims[0])]
        for i in range(1, self.num_subnets):
            subnets.append(Subnetwork(subnet_hidden_dims[i-1][-1] * fifo_len, subnet_hidden_dims[i]))
        self.subnets = nn.ModuleList(subnets)

        # The prediction network will take as input the output of each subnet, as well as the output of the embedding layer:
        prednet_input_size = sum(shd[-1] for shd in subnet_hidden_dims) + embedding_len
        prednet_layers = [nn.Linear(prednet_input_size, pred_hidden_dims[0]), nn.ReLU()]
        for i in range(len(pred_hidden_dims) - 1):
            prednet_layers.extend([nn.Linear(pred_hidden_dims[i], pred_hidden_dims[i+1]), nn.ReLU()])
        prednet_layers.append(nn.Linear(pred_hidden_dims[-1], vocab_size))
        self.prediction_network = nn.Sequential(*prednet_layers)
        self.softmax = nn.LogSoftmax(dim=2)

        self.fifos = None
        self.subnet_out = None # final subnet's output memory element

    def initialize_memory(self, batch_size, device):
        # TOOD: Should I do requires_grad=True here?

        # The first subnet's FIFO is the fifo_len * embedding_len. The other's are the fifo_len * subnet_hidden_dims[i-1][-1]:
        self.fifos = [torch.zeros(batch_size, self.embedding_len * self.fifo_len).to(device)]
        for i in range(1, self.num_subnets):
            self.fifos.append(torch.zeros(batch_size, self.subnet_hidden_dims[i-1][-1] * self.fifo_len, requires_grad=True).to(device))

        self.subnet_out = torch.zeros(batch_size, self.subnet_hidden_dims[-1][-1], requires_grad=True).to(device)        

    def forward(self, input_sequence):
        if self.fifos is None:
            self.initialize_memory(input_sequence.size(0), input_sequence.device)

        embedded = self.embedding_layer(input_sequence) # (batch_size, seq_len, embedding_len)        
        
        # Now we need to calculate all the subnet outputs. They are not all updated at the same time.
        # Each subnet updates at half the rate of the layer below it.
        # If a subnet is supposed to update, then its output will be pushed into
        # its respective FIFO, except for the last subnet's output, which will be
        # saved in a memory element (self.subnet_out). These memories are persistent, so even
        # if a subnet does not update, its previous outputs will be preserved in its FIFO or
        # memory element. These are then used to form the input to the prediction network.
        # Also, we are walking through the sequence one character at a time, but the
        # prediction network needs to see the entire sequence at once. So we will
        # need to accumulate the predicition network's output to feed to the softmax.

        softmax_input = torch.zeros(input_sequence.size(0), self.seq_len, self.vocab_size).to(input_sequence.device)

        for ii in range(self.seq_len):
            # We always embed the current input and push it into the first subnet's input FIFO:
            x = embedded[:, ii, :]
            # Push in new input using the cloing technique from below:
            fifo_shifted = self.fifos[0][:, self.embedding_len:].clone()
            self.fifos[0] = torch.cat((fifo_shifted, x), dim=1)
            #self.fifos[0].roll(-self.embedding_len, dims=1) 
            #self.fifos[0][:, -self.embedding_len:] = x # TODO: this line is a problem

            # Now we walk through the subnets, decide which ones to update, update
            # them, and push their outputs into their respective FIFOs:
            for idx, subnet in enumerate(self.subnets):
                if (ii+1) % (2 ** (idx + 1)) == 0: # Update every 2^(idx+1) inputs
                    shift_by = self.subnet_hidden_dims[idx][-1]

                    # If this is the last subnet...
                    if idx == self.num_subnets - 1:
                        # Save its output in a memory element instead of a FIFO:
                        self.subnet_out = subnet(self.fifos[idx]) # TODO: this line is a problem
                    else:
                        # Otherwise, push its output into the next subnet's input FIFO:
                        #self.fifos[idx+1].roll(-shift_by, dims=1)
                        #self.fifos[idx+1][:, -shift_by:] = subnet(self.fifos[idx]) # TODO: this line is a problem
                        # the above lines cause an inplace modification problem messing up the gradients. So instead:
                        # Compute the output of the subnet
                        subnet_output = subnet(self.fifos[idx])

                        # Clone the relevant slices of the self.fifos tensor
                        fifo_shifted = self.fifos[idx+1][:, :-shift_by].clone()

                        # Concatenate the shifted fifo and subnet output along the desired dimension (1 in your case)
                        new_fifo = torch.cat((fifo_shifted, subnet_output), dim=1)

                        # Update the list of fifos using a list comprehension, replacing only the (idx+1)-th element
                        self.fifos[idx+1] = new_fifo
                        #self.fifos = [f if i != (idx+1) else new_fifo for i, f in enumerate(self.fifos)]


            # Now we need to form the input to the prediction network. This is the concatenation of
            # the the most recent element in each FIFO, as well as the final subnet's output. Note that the size
            # of the prediction network's input is the sum of the sizes of the subnets' outputs, plus
            # the size of the embedding layer's output. The size of the first element in each FIFO is
            # the size of the previous subnet's output:

            # pred_net_input needs to be a tuple of size (batch_size, sum(x[-1] for x in subnet_hidden_dims) + embedding_len)
            # initialize pred_net_input to zeros:
            pred_net_input = torch.zeros(input_sequence.size(0), sum(x[-1] for x in self.subnet_hidden_dims) + self.embedding_len).to(input_sequence.device)
            
            # Now populate pred_net_input.
            # First, populate it with the most recent element in each FIFO:          
            start_idx = 0
            stop_idx = 0
            for ii in range(self.num_subnets):
                if ii == 0:
                    element_size = self.embedding_len
                else:
                    element_size = self.subnet_hidden_dims[ii-1][-1]
                stop_idx += element_size
                pred_net_input[:, start_idx:stop_idx] = self.fifos[ii][:, -element_size:]
                start_idx += element_size
            
            # Now populate pred_net_input with the final subnet's output:
            element_size = self.subnet_hidden_dims[-1][-1]
            stop_idx += element_size            
            pred_net_input[:, start_idx:stop_idx] = self.subnet_out

            # Now we can feed pred_net_input into the prediction network:
            linear_out = self.prediction_network(pred_net_input) # (batch_size, vocab_size)
            # we need to unsqueeze linear_out prior to assigning it to softmax_input:
            softmax_input[:, ii, :] = linear_out # (batch_size, 1, vocab_size)
            

        softmax_out = self.softmax(softmax_input) # (batch_size, seq_len, vocab_size)
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
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.subnet_hidden_dims, args.pred_hidden_dims)


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
        model = CharPredictorMultirateFFN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.fifo_len, args.subnet_hidden_dims, args.pred_hidden_dimss)

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





