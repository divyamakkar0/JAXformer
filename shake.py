import tiktoken 
import numpy as np 

enc = tiktoken.get_encoding("gpt2") 
eot = enc._special_tokens['<|endoftext|>']

def tokenize(text):
    tokens = [eot] 
    tokens.extend(enc.encode(text))
    tokens_np = np.array(tokens).astype(np.uint16)
    idx = int(0.9 * tokens_np.shape[0])
    train = tokens_np[:idx] 
    test = tokens_np[idx:] 

    np.save("./train", train)
    np.save("./test", test)

    return tokens_np

with open('dataset.txt', 'r') as f: 
    txt = f.read() 

tokenize(txt) 
