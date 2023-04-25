import torch
import torch.nn as nn
import numpy as np
import sys

# TODO: fancier weight initialization ("He"?, etc?)

class CharPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_window, hidden_dims):
        super(CharPredictor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Create hidden layers
        hidden_layers = []
        in_dim = embedding_dim * context_window
        for out_dim in hidden_dims:
            hidden_layers.append(nn.Linear(in_dim, out_dim))
            hidden_layers.append(nn.ReLU())
            in_dim = out_dim
        
        self.layers = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(hidden_dims[-1], vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, context_window, embedding_dim)
        x = x.view(x.size(0), -1)  # (batch_size, context_window * embedding_dim)
        x = self.layers(x)  # (batch_size, hidden_dims[-1])
        x = self.output(x)  # (batch_size, vocab_size)
        x = self.softmax(x)  # (batch_size, vocab_size)
        return x



def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss / len(dataloader)}", flush=True)


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

  
def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))


def generate_text(model, initial_context, num_chars, device, vocab_size):
    model.eval()
    model.to(device)
    context = [ord(c) for c in initial_context]
    generated_text = initial_context

    for ii in range(num_chars):
        input_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
        output = model(input_tensor)
        probs = torch.exp(output).cpu().data.numpy().flatten()
        
        # TODO: Force it to only output valid printable ASCII characters in the range > 31 & < 127 (so 32 to 126 endpoint incclusive)
        
        char_index = np.random.choice(np.arange(vocab_size), p=probs)
        generated_text += chr(char_index)
        context.pop(0)
        context.append(char_index)

    return generated_text


def read_corpus(file_path):
    with open(file_path, 'rb') as f:
        #corpus = f.read(2**20)
        corpus = f.read()
    return corpus


def prepare_sequences(corpus, context_window=10):
    corpus_tensor = torch.tensor(list(corpus), dtype=torch.long)
    input_data = []
    target_data = []

    for i in range(len(corpus_tensor) - context_window - 1):
        input_seq = corpus_tensor[i:i + context_window]
        target_char = corpus_tensor[i + context_window]
        input_data.append(input_seq)
        target_data.append(target_char)

    input_data = torch.stack(input_data)
    target_data = torch.tensor(target_data)

    return input_data, target_data


    
