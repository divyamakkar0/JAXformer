import tiktoken
import numpy as np
import os


enc = tiktoken.get_encoding("cl100k_base")

eot = enc._special_tokens["<|endoftext|>"]

def tokenize(text):
    tokens = [eot]
    tokens.extend(enc.encode(text))
    tokens_np = np.array(tokens, dtype=np.uint32)
    return tokens_np

files = []

for i in range(1, 100):
    files.push_back(os.path.join("./train"))