#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

# conda activate thesis


cmd="python ../GatewayExtractionBenchmark.py"
echo "$cmd"
eval "$cmd"
