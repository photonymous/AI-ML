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

# Specify all global constants that aren't arguments:
VOCAB_SIZE     = 256

# Specify defaults for all arguments as ALL_CAPS globals:
MODE           = "train"
SEED_STR       = "Once upon a tim"
NUM_CHARS      = 500
EMBEDDING_LEN  = 16
SEQ_LEN        = 128
WARMUP         = 16
HIDDEN_DIM     = 128
NUM_LAYERS     = 2
NUM_EPOCHS     = 40
BATCH_SIZE     = 1024
MAX_CHARS      = 2**17
CORPUS_FILE    = "/data/training_data/gutenberg_corpus_21MB.txt"
MODEL_FILE     = "trained_rnn.pth"

# Define the command line arguments and assign defaults and format the strings using the globals:
# Note that the arguments can be accessed in code like this: args.mode, args.seed_str, etc.
parser = argparse.ArgumentParser(description='Train or generate text using a character predicting RNN.')
parser.add_argument('--mode', type=str, default=MODE, help='The mode: train or generate (default: %(default)s)')
parser.add_argument('--seed_str', type=str, default=SEED_STR, help='The seed string to use for generating text (default: %(default)s)')
parser.add_argument('--num_chars', type=int, default=NUM_CHARS, help='The number of characters to generate (default: %(default)s)')
parser.add_argument('--embedding_len', type=int, default=EMBEDDING_LEN, help='The embedding length (default: %(default)s)')
parser.add_argument('--seq_len', type=int, default=SEQ_LEN, help='The sequence length (default: %(default)s)')
parser.add_argument('--warmup', type=int, default=WARMUP, help='The warmup (default: %(default)s)')
parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='The hidden dimension (default: %(default)s)')
parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='The number of layers (default: %(default)s)')
parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='The number of epochs (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The batch size (default: %(default)s)')
parser.add_argument('--max_chars', type=int, default=MAX_CHARS, help='The maximum number of characters to read from the corpus file (default: %(default)s)')
parser.add_argument('--corpus_file', type=str, default=CORPUS_FILE, help='The corpus file (default: %(default)s)')
parser.add_argument('--model_file', type=str, default=MODEL_FILE, help='The model file (default: %(default)s)')
args = parser.parse_args()

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
    input_data = []
    target_data = []

    for ii in range(len(corpus_tensor) - seq_len - 1):
        input_seq_start  = ii
        input_seq_stop   = ii + seq_len
        target_seq_start = input_seq_start + 1
        target_seq_stop  = input_seq_stop + 1

        input_seq  = corpus_tensor[input_seq_start:input_seq_stop]
        target_seq = corpus_tensor[target_seq_start:target_seq_stop]
        
        input_data.append(input_seq)
        target_data.append(target_seq)

    input_data  = torch.stack(input_data)
    target_data = torch.stack(target_data)

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

# Create the dataloader:
def create_dataloader(input_sequences, target_sequences, batch_size):
    dataset    = CorpusDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

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
        self.rnn             = nn.RNN(input_size  = embedding_len, 
                                      hidden_size = hidden_dim, 
                                      num_layers  = self.num_layers, 
                                      batch_first = True)
        self.linear          = nn.Linear(hidden_dim, vocab_size)
        self.softmax         = nn.LogSoftmax(dim=2)

    def forward(self, input_sequence):
        # Embed the input sequence:
        embedded = self.embedding_layer(input_sequence)

        # Run the RNN:
        rnn_out, hidden = self.rnn(embedded)

        # Run the linear layer:
        linear_out = self.linear(rnn_out)

        # Run the softmax:
        softmax_out = self.softmax(linear_out)

        return softmax_out

# Train the model:
def train_model(model, dataloader, device, num_epochs, warmup):
    model.train()
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
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
            #   Note that here we need to ignore the first "warmup" characters of both
            #   the outputs and target sequences, because the input sequence is used to
            #   predict the target sequence, and the first outputs of the model
            #   will be garbage because the RNN has not yet seen the warmup characters:
            loss = criterion(outputs[:,warmup:,:], target_sequences[:,warmup:])

            # Backward pass:
            loss.backward()

            # Update the parameters:
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} Batch {batch_idx + 1}/{len(dataloader)} Loss: {loss.item()}", flush=True)    
            
            epoch_loss += loss.item()

            
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss / len(dataloader)}", flush=True)

# Save the model:
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# Load the model:
def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))

# Generate text using the model. The seed_str is the initial context.
#   Note that the length of the seed_str acts as the "warmup". The model will first
#   be fed with the seed_str, one character at a time. It will maintain its
#   internal state between characters. After we've fed the whole seed_str, its very
#   next output will be the first character of the generated text. We will then feed
#   the character back into the model, and it will output
#   the next character, and so on. We will do this for num_chars characters.
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
        model = CharPredictorRNN(VOCAB_SIZE, args.embedding_len, args.seq_len, args.hidden_dim, args.num_layers)

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





