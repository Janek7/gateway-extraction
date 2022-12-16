#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# conda activate thesis

for CONTEXT in 0 1 2 3; do
  cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --mode=context_labels_n_gram --context_size=$CONTEXT --n_gram=0"
  echo "$cmd"
  eval "$cmd"
done