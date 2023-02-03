#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

# conda activate thesis


cmd="python ../RelationClassificationBenchmark.py"
echo "$cmd"
eval "$cmd"
