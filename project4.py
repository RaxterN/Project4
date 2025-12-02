import torch
from torch import nn as nn
from MiniTransformer import MiniTransformer
from Tokenizer import Tokenizer

#Load the text from file
with open("./data.txt", "r", encoding="utf-8") as file:
    plaintext = file.read()

#Instantiate he tokenizer and tokenize the raw text
tokenizer = Tokenizer(plaintext)


# To do: 
# Tokenize the data with tokenizer.encode(...)...
# Create the actual model etc. 
# Training loop
# Interaction loop
# Save and load model weights
