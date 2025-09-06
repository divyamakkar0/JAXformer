import os
import shutil
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from google.cloud.storage import Client, transfer_manager
import argparse
import ray

# init ray in the cluster mode
ray.init(address="auto")

# constants for splits and multiprocessing
TEST_SPLIT = 350
BUCKET_NAME = "ray_jaxformer"
BATCH_SIZE = 512
WORKERS = int(os.cpu_count())
nprocs = max(1, int(os.cpu_count() / 1.5))

# other constants for dataset processing
local_dir = "data_dir"
remote_name = "sample-350BT"
shard_size = int(1e8)

# gcp storage client and bucket
storage_client = Client()
bucket = storage_client.bucket(BUCKET_NAME)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# set up argument parser to check if --continue flag is given
def setup_argument_parser():
  parser = argparse.ArgumentParser(description='Process the 350BT dataset')
  parser.add_argument('--continue', dest='continue_processing', action='store_true',
            help='Continue processing from a checkpoint')
  parser.set_defaults(continue_processing=False)
  return parser

parser = setup_argument_parser()
args = parser.parse_args()
continue_processing = args.continue_processing
checkpoint_to_resume = None
shard_to_resume = 0
skip_number = 0

# if a --continue flag is given, pull latest checkpoint name from gcp bucket called checkpoints
if continue_processing:
  # pull latest checkpoint name from gcp bucket called checkpoints
  blobs = bucket.list_blobs(prefix="checkpoints/")
  checkpoint_blobs = [b for b in blobs if str(b.name).endswith(".txt")]
  if not checkpoint_blobs:
    continue_processing = False
  else:
    latest_checkpoint = max(checkpoint_blobs, key=lambda b: b.updated)
    checkpoint_to_resume = latest_checkpoint.name[len("checkpoints/"):-4]  # remove 'checkpoints/' prefix and '.txt' suffix
    shard_to_resume, skip_number = map(int, (latest_checkpoint.download_as_bytes().decode('utf-8')).split(':'))

# ------------------------------------------

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

# init the tokenizer
enc = tiktoken.encoding_for_model("gpt-4") # 'cl100k_base'
eot = enc._special_tokens['<|endoftext|>'] # end of text token

# tokenize function with ray remote decorator
@ray.remote
def tokenize(doc):
  doc_id_return = doc['id']
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc["text"]))
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
  tokens_np_uint32 = tokens_np.astype(np.uint32)
  return tokens_np_uint32, doc_id_return

def write_datafile(filename, tokens_np):
  np.save(filename, tokens_np)

# function to upload files to gcp bucket using transfer manager
def upload_file(split):
  def upload_many_blobs_with_transfer_manager(split, filenames, source_directory="", workers=8):

    blob_names = [split + name for name in filenames]

    blob_file_pairs = [(os.path.join(source_directory, f), bucket.blob(b)) for f, b in zip(filenames, blob_names)]

    results = transfer_manager.upload_many(
      blob_file_pairs, skip_if_exists=True, max_workers=workers, worker_type=transfer_manager.THREAD
    )

  FILE_NAMES = os.listdir(DATA_CACHE_DIR)
  upload_many_blobs_with_transfer_manager(split, FILE_NAMES, DATA_CACHE_DIR, WORKERS)
  for file in FILE_NAMES:
    full_path = DATA_CACHE_DIR + '/' + file
    os.remove(full_path)

# function to upload checkpoints to gcp bucket and remove local copies
def upload_checkpoint():
  checkpoint_files = os.listdir(checkpoint_dir)
  for filename in checkpoint_files:
    blob = bucket.blob(f"checkpoints/{filename}")
    blob.upload_from_filename(os.path.join(checkpoint_dir, filename))
  for filename in checkpoint_files:
    os.remove(os.path.join(checkpoint_dir, filename))

# skip to the previous checkpoint (zero by default)
fw.skip(skip_number)
shard_index = shard_to_resume + 1 if continue_processing else 0

# variables to keep track of tokens in the current shard
all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
token_count = 0
progress_bar = None
doc_iter = iter(fw)

while True:
    batch = []
    try:
      for _ in range(BATCH_SIZE):
        batch.append(next(doc_iter))
    except StopIteration:
      pass
    
    if not batch:
      break
    
    # get the tokenized results from ray
    futures = [tokenize.remote(doc) for doc in batch]
    results = ray.get(futures)
    
    for tokens, doc_id in results:
      skip_number += 1

      # if the current document fits in the current shard
      if token_count + len(tokens) < shard_size:
        
        # simply append tokens to current shard
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
      
        # update progress bar
        if progress_bar is None:
          progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}", dynamic_ncols=True)
        progress_bar.update(len(tokens))
      
      else:
      
        # save a checkpoint for resuming later
        checkpoint_filename = os.path.join(checkpoint_dir, f"{doc_id}.txt")
        with open(checkpoint_filename, "w") as f:
          f.write(str(shard_index) + ':' + str(skip_number))

        # write the current shard and start a new one
        if shard_index >= 0 and shard_index < TEST_SPLIT:
          split = 'test/'
          shard_index_number = shard_index
        else:
          split = 'train/'
          shard_index_number = shard_index - TEST_SPLIT
        split_name = split[:-1]

        filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_{shard_index_number:04d}")

        # split the document into whatever fits in this shard; the remainder goes to next one
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

        write_datafile(filename, all_tokens_np)
        upload_file(split)
        upload_checkpoint()

        # update shard index and reset progress bar
        shard_index += 1
        progress_bar = None
        
        # populate the next shard with the leftovers of the current doc
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder

# write any remaining tokens as the last shard
if token_count != 0:
  if shard_index >= 0 and shard_index < TEST_SPLIT:
    split = 'test/'
    shard_index_number = shard_index
  else:
    split = 'train/'
    shard_index_number = shard_index - TEST_SPLIT
  split_name = split[:-1]
    
  filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_{shard_index_number:04d}")

  write_datafile(filename, all_tokens_np[:token_count])
  upload_file(split)
  upload_checkpoint()


# clean up directory after function terminates
if os.path.exists(DATA_CACHE_DIR):
  shutil.rmtree(DATA_CACHE_DIR)
