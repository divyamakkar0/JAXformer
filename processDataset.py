import tiktoken
import numpy as np
import threading
import os
import time
import asyncio
import argparse


MAX_THREADS = 16
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


async def tokenize(file_path, save_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = [eot]
    tokens.extend(enc.encode(text))
    tokens_np = np.array(tokens, dtype=np.uint32)
    base_path = os.path.abspath(save_path)
    out_file = os.path.join(base_path, file_path.split("/")[-1] + ".npy")
    np.save(out_file, tokens_np)


async def tokenize_fn(files, save_path):
    while len(files) > 0:
        await tokenize(files.pop(0), save_path)


def thread_fn(files, save_path):
    asyncio.run(tokenize_fn(files, save_path))


def main(file_path, save_path):
    files = [[] for _ in range(MAX_THREADS)]

    base_path = os.path.abspath(file_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    print(base_path)
    files_in_dir = os.listdir(base_path)
    for idx, file_name in enumerate(files_in_dir):
        current_path = os.path.join(base_path, file_name)
        files[idx % MAX_THREADS].append(current_path)

    threads = []
    for i in range(MAX_THREADS):
        thread = threading.Thread(target=thread_fn, args=(files[i], save_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
        threads.remove(thread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    file_path = os.path.abspath(args.file_path)
    save_path = file_path + "Shards"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        for file in os.listdir(save_path):
            os.remove(os.path.join(save_path, file))
    start = time.time()
    main(file_path, save_path)
    end = time.time()
    print(f"all files are tokenized in {end - start} seconds")
