# This will be a character predicting RNN. It will be trained using a large corpus of
# raw text, and will be able to generate new text based on a seed string. It will process
# one character at a time, and will output a probability distribution over the next
# character. The output will be a softmax over the vocabulary size (256 characters).
# It will use a standard RNN, not LSTM or GRU. The input will be embedded, with a
# specifiable embedding length. The sequence length will be specifiable. The hidden
# layer will be specifiable. The number of hidden layers will be specifiable. The
# number of epochs will be specifiable. The batch size will be specifiable. 
# It will all be in one file, and a command line argument will specify "trian"
# or "generate". 

# Import the libraries we will need:
import torch
import torch.nn as nn
import numpy as np
import sys
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import datetime
# import a learning rate scheduler:
from torch.optim.lr_scheduler import ExponentialLR

# Specify all global constants that aren't arguments:
VOCAB_SIZE     = 256
LEARNING_RATE  = 0.001
LR_GAMMA       = 0.999

# Specify defaults for all arguments as ALL_CAPS globals:
MODE           = "generate"
#                 0        1         2         3         4         5         6         7
#                 1234567890123456789012345678901234567890123456789012345678901234567890
SEED_STR       = "In the morning, he had gone down to the river (to wash his cloth"
NUM_CHARS      = 500
EMBEDDING_LEN  = 32
SEQ_LEN        = 256
WARMUP         = 64
HIDDEN_DIM     = 256
NUM_LAYERS     = 5
NUM_EPOCHS     = 1000
BATCH_SIZE     = 128
MAX_CHARS      = 2**24
CORPUS_FILE    = "/data/training_data/gutenberg_corpus_21MB.txt"
MODEL_FILE     = "trained_rnn.pth"

# Define the command line arguments and assign defaults and format the strings using the globals:
# Note that the arguments can be accessed in code like this: args.mode, args.seed_str, etc.
parser = argparse.ArgumentParser(description='Train or generate text using a character predicting RNN.')
parser.add_argument('--mode',          type=str, default=MODE, help='The mode: train or generate (default: %(default)s)')
parser.add_argument('--seed_str',      type=str, default=SEED_STR, help='The seed string to use for generating text (default: %(default)s)')
parser.add_argument('--num_chars',     type=int, default=NUM_CHARS, help='The number of characters to generate (default: %(default)s)')
parser.add_argument('--embedding_len', type=int, default=EMBEDDING_LEN, help='The embedding length (default: %(default)s)')
parser.add_argument('--seq_len',       type=int, default=SEQ_LEN, help='The sequence length (default: %(default)s)')
parser.add_argument('--warmup',        type=int, default=WARMUP, help='The warmup (default: %(default)s)')
parser.add_argument('--hidden_dim',    type=int, default=HIDDEN_DIM, help='The hidden dimension (default: %(default)s)')
parser.add_argument('--num_layers',    type=int, default=NUM_LAYERS, help='The number of layers (default: %(default)s)')
parser.add_argument('--num_epochs',    type=int, default=NUM_EPOCHS, help='The number of epochs (default: %(default)s)')
parser.add_argument('--batch_size',    type=int, default=BATCH_SIZE, help='The batch size (default: %(default)s)')
parser.add_argument('--max_chars',     type=int, default=MAX_CHARS, help='The maximum number of characters to read from the corpus file (default: %(default)s)')
parser.add_argument('--corpus_file',   type=str, default=CORPUS_FILE, help='The corpus file (default: %(default)s)')
parser.add_argument('--model_file',    type=str, default=MODEL_FILE, help='The model file (default: %(default)s)')
args = parser.parse_args()

# Define the model:
class CharPredictorRNN(nn.Module):    
    def __init__(self, vocab_size, embedding_len, seq_len, hidden_dim, num_layers):
        super(CharPredictorRNN, self).__init__()

        self.vocab_size    = vocab_size
        self.embedding_len = embedding_len
        self.seq_len       = seq_len
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        
        self.embedding_layer = nn.Embedding(vocab_size, embedding_len)
        self.rnn             = nn.GRU(input_size  = embedding_len, 
                                      hidden_size = hidden_dim, 
                                      num_layers  = self.num_layers, 
                                      batch_first = True)
        self.linear          = nn.Linear(hidden_dim, vocab_size)
        self.softmax         = nn.LogSoftmax(dim=2)

    def forward(self, input_sequence, hidden):
        # Embed the input sequence:
        #  embedded is a tensor of size (batch_size, seq_len, embedding_len).
        embedded = self.embedding_layer(input_sequence)

        # Run the RNN. When the RNN runs on the input_sequence, it will return
        #   rnn_out, which is a tensor of size (batch_size, seq_len, hidden_dim).
        #   hidden is a tensor of size (num_layers, batch_size, hidden_dim).
        rnn_out, hidden = self.rnn(embedded, hidden)

        # Run the linear layer. 
        #   linear_out is a tensor of size (batch_size, seq_len, vocab_size).
        linear_out = self.linear(rnn_out)

        # Run the softmax:
        #   softmax_out is a tensor of size (batch_size, seq_len, vocab_size).
        softmax_out = self.softmax(linear_out)

        return softmax_out, hidden
    
def init_hidden(batch_size, hidden_dim, num_layers):
    return torch.zeros(num_layers, batch_size, hidden_dim)

# Preprocess the corpus data as raw 8-bit binary data:
def read_corpus(file_path, max_chars=None):
    with open(file_path, "rb") as f:
        corpus = f.read(max_chars)
    return corpus

# Prepare the input and target sequences. Remember,
# the input data is unsigned 8-bit values. Every character in the
# input sequence will have a corresponding target character.
def prepare_sequences(corpus, seq_len):
    corpus_tensor = torch.tensor(list(corpus), dtype=torch.long)
    
    # First, we need to figure out how many batches we can create:
    num_batches = len(corpus_tensor) // seq_len

    # Now we need to figure out how many characters we need to drop from the end of the corpus tensor
    # so that we can create an input sequence of length seq_len:
    num_chars_to_drop = len(corpus_tensor) - (num_batches * seq_len)

    # Now we need to drop those characters from the end of the corpus tensor:
    if num_chars_to_drop > 0:
        corpus_tensor = corpus_tensor[:-num_chars_to_drop]

    # Now we need to reshape the corpus tensor into a tensor of size (num_batches, seq_len),
    # and we'll use view() so that we don't have to copy the data:
    corpus_tensor = corpus_tensor.view(num_batches, seq_len)

    # Now we need to create the input sequences. Each input sequence is just a little piece of the corpus.
    # We can't quite use all of the sequence, since the target data predicts the second character through
    # the last character, so the input data has to be the first character to the second to last character.
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

            # Initialize the hidden state:
            hidden = init_hidden(args.batch_size, args.hidden_dim, args.num_layers).to(device)

            # Forward pass:
            outputs, hidden = model(input_sequences, hidden)

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
        
        #print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.5f} ETA: {end_time.strftime("%H:%M:%S")}", flush=True)
        #print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.5f} ETA: {end_time.strftime('{{%H:%M:%S}}')} LR:{old_lr:.5f}->{new_lr:.5f}", flush=True)
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
    hidden = init_hidden(1, args.hidden_dim, args.num_layers).to(device)
    context = [ord(c) for c in seed_str]
    
    # Feed one character at a time to the model: (warmup)
    output = None
    for ii in range(len(context)):
        input_tensor = torch.tensor(context[ii], dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)

        # Run the model, but ignore the output for now
        output, hidden = model(input_tensor, hidden)
    
    # Now we will feed the final output of the model back into the model, one character at a time:
    generated_text = seed_str
    for ii in range(num_chars):
        # Sample a character from the output distribution:
        probs = torch.exp(output).cpu().data.numpy().flatten()
        char_idx = np.random.choice(vocab_size, p=probs)
        generated_text += chr(char_idx)

        # Feed the character back into the model:
        input_tensor = torch.tensor(char_idx, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)
        output, hidden = model(input_tensor, hidden)

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
        model = CharPredictorRNN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.hidden_dim, args.num_layers)

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
        model = CharPredictorRNN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.hidden_dim, args.num_layers)

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

# Here's how to run it:
#    $ python char_predictor_rnn.py --mode train --num_epochs 40 --batch_size 1024 --max_chars 2**17 --model_file trained_rnn.pth
#    $ python char_predictor_rnn.py --mode generate --seed_str "Once upon a tim" --num_chars 500 --model_file trained_rnn.pth





