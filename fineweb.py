import datasets
import os
import numpy as np
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
fw = datasets.load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=False,
    num_proc=4
)

out_folder_train = os.path.abspath("./fineWebTrain")
out_folder_val = os.path.abspath("./fineWebVal")


length = fw['num_rows']
val_tokens = int(length * 0.05)
idx = np.random.choice(length, val_tokens, replace=False)

NUM_THREADS = 16



for i in idx:
  fw[i]['text']

for i, batch in enumerate(fw):
  batch['text']
  break
