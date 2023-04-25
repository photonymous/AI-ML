from torch.utils.data import DataLoader, TensorDataset
from char_predictor import *

vocab_size = 256
embedding_dim = 64
context_window = 64
hidden_dims = [1024,1024,1024,1024,1024,1024,1024,1024,1024,1024]

batch_size = 8192
num_epochs = 200

# Read the corpus and prepare the input and target sequences
corpus = read_corpus("corpus.txt")
input_sequences, target_sequences = prepare_sequences(corpus, context_window)

# Create a DataLoader for the input and target sequences
dataset = TensorDataset(input_sequences, target_sequences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Initialize the model, loss function, and optimizer
model = CharPredictor(vocab_size, embedding_dim, context_window, hidden_dims)
total_parameters = sum(param.numel() for param in model.parameters())
criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print(f"Total number of trainable parameters: {total_parameters}", flush=True)

# Train the model
train_model(model, dataloader, criterion, optimizer, device, num_epochs)

# Save the trained model
save_model(model, "trained_model.pth")

