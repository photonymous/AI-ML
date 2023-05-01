from char_predictor import *


vocab_size = 256
embedding_dim = 16
context_window = 16
hidden_dims = [32,64,128]
#############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharPredictor(vocab_size, embedding_dim, context_window, hidden_dims)


# Load the trained model
load_model(model, "trained_model.pth", device)

# Generate text using the trained model
#                  0        1         2         3         4         5         6     
#                  123456789012345678901234567890123456789012345678901234567890123456789
initial_context = "have paid for th"
num_chars = 500
generated_text = generate_text(model, initial_context, num_chars, device, vocab_size)
print(generated_text)




