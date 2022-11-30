#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis


cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-11 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --use_synonyms=True --mode=concat --context_size=1 --n_gram=1"
echo "$cmd"
eval "$cmd"
