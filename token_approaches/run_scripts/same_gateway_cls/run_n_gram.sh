#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# conda activate thesis

for N_GRAM in 0 1 2 3 4 5; do
  cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
        --routine=cv --folds=5 --mode=n_gram --n_gram=$N_GRAM"
  echo "$cmd"
  eval "$cmd"
done
