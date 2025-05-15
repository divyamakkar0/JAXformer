import tiktoken
import numpy as np
import threading
import os
import time
import asyncio

enc = tiktoken.get_encoding("cl100k_base")
eot = enc._special_tokens["<|endoftext|>"]

#async version

# async def tokenize(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()
#     tokens = [eot]
#     tokens.extend(enc.encode(text))
#     tokens_np = np.array(tokens, dtype=np.uint32)
#     base_path = os.path.abspath("./trainSetShards")
#     out_file = os.path.join(base_path, file_path.split("/")[-1] + ".npy")
#     np.save(out_file, tokens_np)

# async def main():
#     files = []

#     base_path = os.path.abspath("./trainSet")
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)
#     print(base_path)
#     for i in range(1, 100):
#         files.append(os.path.join(base_path, f"news.en-00{i:03d}-of-00100"))

#     max_threads = 16
#     while len(files) > 0:
#         await tokenize(files.pop(0))

# if __name__ == "__main__":
#     if not os.path.exists("./trainSetShards"):
#         os.makedirs("./trainSetShards")
#     else:
#         for file in os.listdir("./trainSetShards"):
#             os.remove(os.path.join("./trainSetShards", file))
#     start = time.time()
#     asyncio.run(main())
#     end = time.time()
#     print(f"all files are tokenized in {end - start} seconds")


# threading version
# def tokenize(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()
#     tokens = [eot]
#     tokens.extend(enc.encode(text))
#     tokens_np = np.array(tokens, dtype=np.uint32)
#     base_path = os.path.abspath("./trainSetShards")
#     out_file = os.path.join(base_path, file_path.split("/")[-1] + ".npy")
#     np.save(out_file, tokens_np)

# def main():

#     files = []

#     base_path = os.path.abspath("./trainSet")
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)
#     print(base_path)
#     for i in range(1, 100):
#         files.append(os.path.join(base_path, f"news.en-00{i:03d}-of-00100"))

#     max_threads = 16
#     threads = []
#     while len(files) > 0:
#         for i in range(min(max_threads, len(files))):
#             file_path = files.pop(0)
#             thread = threading.Thread(target=tokenize, args=(file_path,))
#             threads.append(thread)
#             thread.start()

#         for thread in threads:
#             thread.join()
#             threads.remove(thread)

# if __name__ == "__main__":
#     if not os.path.exists("./trainSetShards"):
#         os.makedirs("./trainSetShards")
#     else:
#         for file in os.listdir("./trainSetShards"):
#             os.remove(os.path.join("./trainSetShards", file))
#     start = time.time()
#     main()
#     end = time.time()
#     print(f"all files are tokenized in {end - start} seconds")