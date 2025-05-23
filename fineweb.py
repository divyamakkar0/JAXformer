import datasets
import os

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

for i, batch in enumerate(fw):
  batch['text']
  break