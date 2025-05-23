#!/bin/bash

pip install datasets
python3 <<EOF
import datasets
fw = datasets.load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split=None,
    streaming=False,
    num_proc=4
)
EOF
pip uninstall -y datasets