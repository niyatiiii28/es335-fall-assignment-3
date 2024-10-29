import json
import streamlit as st
import torch
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Load JSON files
with open('Ques_1/streamlit/word_to_idx.json', 'r') as f:
    stoi = json.load(f)

with open('Ques_1/streamlit/idx_to_word.json', 'r') as f:
    itos = json.load(f)

# Ensure you have the <unk> token defined
if "<unk>" not in stoi:
    stoi["<unk>"] = len(stoi)  # Assign it to the next index

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model class definition
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, context_len, activation):
        super(NextWordMLP, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(context_len * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        self.activation = activation  # Store activation function type

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)  # Flatten the input
        if self.activation == "relu":
            x = F.relu(self.lin1(x))
        else:
            x = F.tanh(self.lin1(x))
        x = self.lin2(x)
        return x

# Function to load model
def load_model(embedding_dim, context_len, activation):
    model_path = f"model_{activation}_{embedding_dim}_{context_len}.pth"  # Temporary local file
    model_url = f"https://github.com/Manasa2810/es335-fall-assignment-3/releases/download/v1.0/{model_path}"  # Replace with your URL

    # Initialize model with the original vocab size
    original_vocab_size = 16814  # Set this to the correct size
    model = NextWordMLP(original_vocab_size, embedding_dim, 1024, context_len, activation).to(device)

    try:
        # Try to load the model directly
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"{model_path} not found. Downloading from GitHub...")
        download_model(model_url, model_path)
        checkpoint = torch.load(model_path, map_location=device)

    # Print keys in the checkpoint to understand its structure
    print("Checkpoint keys:", checkpoint.keys())

    # Load state_dict or the parameters directly depending on the checkpoint structure
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)  # Load directly if no 'state_dict' key

    model.eval()
    
    return model

# Streamlit user inputs
st.title("Next Word Prediction")

# Dropdowns for user selections
embedding_dim = st.selectbox("Select Embedding Dimension", [32, 64])
context_len = st.slider("Select Context Length", min_value=5, max_value=15, step=5)
activation = st.selectbox("Select Activation Function", ["relu", "tanh"])
k = st.number_input("Enter number of words to predict (k)", min_value=1, step=1)

# User input for text prompt
input_text = st.text_input("Enter your text:")

# Load selected model
model = load_model(embedding_dim, context_len, activation)

# Helper function for text preprocessing
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9 \.]", "", text).lower()  # Remove special chars and lowercase
    words = text.split()
    return words

# Generate predictions
import random

import random

import random
import torch

def predict_next_words(model, input_text, k=3, top_k=10, temperature=1.0, seed=None):
    # Set the random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    words = preprocess_text(input_text)[-context_len:]  # Get last `context_len` words
    words_idx = [stoi.get(word, stoi["<unk>"]) for word in words]
    x = torch.tensor([words_idx], dtype=torch.long).to(device)

    generated_words = []
    vocabulary_list = list(stoi.keys())  # Vocabulary list for fallback options
    used_words = set()

    for _ in range(k):
        with torch.no_grad():
            logits = model(x) / temperature
            probs = F.softmax(logits, dim=-1).squeeze(0)

            # Limit to the top_k words for diversity and sample
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            top_k_probs /= top_k_probs.sum()  # Normalize probabilities
            next_word_idx = torch.multinomial(top_k_probs, 1).item()
            next_word = itos.get(top_k_indices[next_word_idx].item(), "<unk>")

            # Fallback to a random word from vocabulary if <unk> or already used
            while next_word == "<unk>" or next_word in used_words:
                next_word = random.choice(vocabulary_list)

            generated_words.append(next_word)
            used_words.add(next_word)  # Track used words to prevent repetition

            # Update input tensor
            next_word_idx_tensor = torch.tensor([[top_k_indices[next_word_idx]]], dtype=torch.long).to(device)
            x = torch.cat([x[:, 1:], next_word_idx_tensor], dim=1)

    return " ".join(generated_words)




# Run prediction and display result
if st.button("Predict"):
    if input_text:
        next_words = predict_next_words(model, input_text, k)
        st.write(f"Predicted next {k} word(s): {next_words}")
    else:
        st.write("Please enter text for prediction.")
