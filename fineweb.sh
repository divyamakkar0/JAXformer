#!/bin/bash

# script to get the fineweb-edu-10bt-for-gpt2 dataset off kaggle

script_dir=$(dirname "$(realpath "$0")")

curl -L -o "$script_dir/fineweb-edu-10bt-for-gpt2.zip"\
    https://www.kaggle.com/api/v1/datasets/download/minhthiennguyen/fineweb-edu-10bt-for-gpt2

unzip "$script_dir/fineweb-edu-10bt-for-gpt2.zip" -d "$script_dir/fineweb-edu-10bt-for-gpt2"

rm "$script_dir/fineweb-edu-10bt-for-gpt2.zip"

rm "$script_dir/fineweb-edu-10bt-for-gpt2/SHA256SUMS"
rm "$script_dir/fineweb-edu-10bt-for-gpt2/SHA512SUMS"