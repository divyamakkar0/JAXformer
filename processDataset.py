import tiktoken
import numpy as np


enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(text):
    tokens = [eot]
    tokens.extend(enc.encode(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


with open("shake.txt", "r") as f:
    txt = f.read()

tokens = tokenize(txt)
np.save("tokens.npy", tokens)
