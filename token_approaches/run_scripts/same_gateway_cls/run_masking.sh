#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

# conda activate thesis

for MASKING in "not" "dummy" "single_mask" "multi_mask"; do
  cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --activity_masking=$MASKING --mode=context_n_gram --context_size=0 --n_gram=0"
  echo "$cmd"
  eval "$cmd"
done