#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

# conda activate thesis

# make sure correct method is about to be executed at the bottom of the file
cmd="python ../activity_relation_dataset_preparation.py"
echo "$cmd"
eval "$cmd"