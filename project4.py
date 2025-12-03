import torch
from torch import nn as nn
import torch.nn.functional as fun
from MiniTransformer import TinyLM
from Tokenizer import Tokenizer

#Load the text from file
with open("./data.txt", "r", encoding="utf-8") as file:
    plaintext = file.read()

#Instantiate he tokenizer and tokenize the raw text
tokenizer = Tokenizer(plaintext)
vocab_size = tokenizer.vocab_size

#Encode tokens to id's
token_ids = tokenizer.encode(plaintext)
data = torch.tensor(token_ids)
print(f"Tokens in dataset: {len(data)}")
print(f"Vocab size: {vocab_size}")

#Split data test / train
split = 0.8
split_point = int(len(data) * split)
training_data = data[:split_point]
testing_data = data[split_point:]

#Build the model
model = TinyLM(vocab_size=vocab_size, d_model=64, n_heads=2, num_layers=2, dim_feedforward=256, max_seq_len=128)
optimizer = torch.optim.Adam(model.parameters())

#Training loop
block_size = 64

train = training_data
num_examples = len(train) - block_size - 1

inputs_list = []
targets_list = []

for i in range(num_examples):
    x = train[i : i + block_size]
    y = train[i + 1 : i + 1 + block_size]
    inputs_list.append(x)
    targets_list.append(y)

inputs = torch.stack(inputs_list)
targets = torch.stack(targets_list)

batch_size = 32
epochs = 8

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(inputs.size(0))  # shuffle the examples
    inputs_shuffled = inputs[perm]
    targets_shuffled = targets[perm]

    total_loss = 0.0

    for start in range(0, inputs.size(0), batch_size):
        end = start + batch_size
        xb = inputs_shuffled[start:end]
        yb = targets_shuffled[start:end]

        logits = model(xb)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = yb.view(-1)

        optimizer.zero_grad()
        loss = fun.cross_entropy(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (inputs.size(0) / batch_size)
    print(f"Epoch {epoch}: avg loss = {avg_loss:.3f}")



