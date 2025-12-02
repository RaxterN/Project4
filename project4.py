import torch
from torch import nn as nn
from MiniTransformer import MiniTransformer

#Load the text from file
with open("./data.txt", "r", encoding="utf-8") as file:
    plaintext = file.read()

