import torch
from torch import nn as nn
from torch import optim as optim
import torch.utils.data as data
from MiniTransformer import MiniTransformer

#Load the text from file
with open("./data.txt", "r", encoding="utf-8") as file:
    text = file.read()

#Create word-level vocabulary
words = text.split()
print(f"Token count: {len(words)}")
vocab = sorted(list(set(words))) #gets only the unique words
stoi = {word: i for i, word in enumerate(vocab)} #string to integer dictionary
itos = {i: word for word, i in stoi.items()} #integer to string dictionary

#Tokenize and encode
encoded = [stoi[w] for w in words]

#Split into input, target pairs
seq_length = 5
inputs = []
targets = []
for i in range(len(encoded) - seq_length):
    inputs.append(encoded[i:i+seq_length])
    targets.append(encoded[i+1:i+seq_length+1])

#Convert to tensor objects for pytorch to work with
inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

#Create the model
model = MiniTransformer(vocab_size = len(vocab))
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

#Training loop
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    probs = model(inputs)  
    log_probs = torch.log(probs + 1e-9)   

    log_probs_flat = log_probs.view(-1, len(vocab))
    targets_flat = targets.view(-1)

    loss = loss_func(log_probs_flat, targets_flat)

    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.3f}")

def generate(prompt, max_new_tokens=30):
    prompt_tokens = []
    for token in prompt.split():
        if token in stoi:
            prompt_tokens.append(stoi[token])
    
    x = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):

        probs = model(x)
        last_probs = probs[0, -1]

        next_id = torch.argmax(last_probs).item()
        next_token = torch.tensor([[next_id]], dtype=torch.long)
        x = torch.cat([x, next_token], dim=1)
        prompt_tokens.append(next_id)

    return " ".join(itos[i] for i in prompt_tokens)

####################################### Interaction Loop
#Need to change this
prompt = "Back in my day"
print("Prompt:", prompt)
print("Generation:", generate(prompt, max_new_tokens=20))
