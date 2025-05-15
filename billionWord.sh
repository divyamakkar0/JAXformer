# !/bin/bash

# Script to prepare billion word shards

if [ -f "training-monolingual.tgz" ]; then
    echo "File already downloaded."
else
    wget http://statmt.org/wmt11/training-monolingual.tgz
fi

git clone https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark.git
cd 1-billion-word-language-modeling-benchmark
tar --extract -v --file ../statmt.org/tar_archives/training-monolingual.tgz --wildcards training-monolingual/news.20??.en.shuffled
bash ./scripts/get-data.sh
mv ./training-monolingual.tokenized.shuffled/ ../trainSet/
mv ./heldout-monolingual.tokenized.shuffled/ ../valSet/
cd ../
rm -rf 1-billion-word-language-modeling-benchmark
