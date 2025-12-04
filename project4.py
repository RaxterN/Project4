import torch
import os
from torch import nn as nn
import torch.nn.functional as fun
from MiniTransformer import TinyLM
from Tokenizer import Tokenizer

def Generate(tokenizer, model, prompt, response_length):
    device = torch.device("cpu")

    #Encode prompt, check that prompt includes known tokens
    encoded = tokenizer.encode(prompt)
    if len(encoded) == 0:
        return "(Prompt contains no known tokens.\n)"
    input_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device,).unsqueeze(0)

    for _ in range(response_length):
        if input_tokens.size(1) > model.max_seq_len:
            input_cond = input_tokens[:, -model.max_seq_len:]
        else:
            input_cond = input_tokens

        #Forward pass
        logits = model(input_cond)

        #Get only logits for last position in seqence
        next_logits = logits[:, -1, :]

        #Get probabilities
        probs = fun.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        input_tokens = torch.cat([input_tokens, next_token], dim=1)  # [1, T+1]

    generated_tokens = input_tokens[0].tolist()
    return tokenizer.decode(generated_tokens)

def LoadFromWeights(data_filepath, weights_filename):
    #Load the text from file
    with open(data_filepath, "r", encoding="utf-8") as file:
        plaintext = file.read()

    #Instantiate he tokenizer and tokenize the raw text
    tokenizer = Tokenizer(plaintext)
    vocab_size = tokenizer.vocab_size

    #Build the model
    model = TinyLM(vocab_size=vocab_size, d_model=64, n_heads=2, num_layers=2, dim_feedforward=256, max_seq_len=128)

    #Load model weights
    state = torch.load(weights_filename, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    #Prompt and generate loop
    print("Entering prompting mode, say \"Exit\" to exit:")
    while True:
        prompt = input("> ")
        prompt = prompt.lower()
        if prompt == "exit": break

        response = Generate(tokenizer, model, prompt, response_length=25)
        print(f"> {response}")

def Train(data_filepath, weights_filename, batch_size, epochs):
    #Load the text from file
    with open(data_filepath, "r", encoding="utf-8") as file:
        plaintext = file.read()

    #Instantiate he tokenizer and tokenize the raw text
    tokenizer = Tokenizer(plaintext)
    vocab_size = tokenizer.vocab_size

    #encode tokens to id's
    token_ids = tokenizer.encode(plaintext)
    data = torch.tensor(token_ids)
    print(f"Tokens in dataset: {len(data)}")
    print(f"Vocab size: {vocab_size}")

    #Build the model
    model = TinyLM(vocab_size=vocab_size, d_model=64, n_heads=2, num_layers=2, dim_feedforward=256, max_seq_len=128)
    optimizer = torch.optim.Adam(model.parameters())

    #Training loop
    block_size = 64

    num_examples = len(data) - block_size - 1

    inputs_list = []
    targets_list = []

    for i in range(num_examples):
        x = data[i : i + block_size]
        y = data[i + 1 : i + 1 + block_size]
        inputs_list.append(x)
        targets_list.append(y)

    inputs = torch.stack(inputs_list)
    targets = torch.stack(targets_list)

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

    #Save weights for each training run to file
    # after your training loop
    torch.save(model.state_dict(), f"{weights_filename}.pth")
    print(f"Saved weights to {weights_filename}.pth")

def PrintMenu():
    print("[1] - New Training Run")
    print("[2] - Load Existing Weights")
    print("[Exit] - Exit Program")

# User interaction loop ################################################################
while True:
    PrintMenu()
    user_input = input("> ")
    if user_input.lower() == "exit":
        print("Exiting Program.")
        break
    elif user_input == "1":
        print("Enter the weights filename.")
        weights_filename = input("> ")
        Train("./data2.txt", weights_filename, 32, 2)
        print("\n Training run complete.")
    elif user_input == "2":
        print("Enter weights filename to load: (do not include file extension!)")
        weights_filename = input("> ")
        if os.path.exists(f"./{weights_filename}.pth"):
            print("Loading Weights.\n")
            LoadFromWeights("./data2.txt", f"./{weights_filename}.pth")
        elif not os.path.exists(f"./{weights_filename}.pth"):
            print("File not found.\n")


