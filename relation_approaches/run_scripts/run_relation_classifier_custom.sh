#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

cmd="python ../RelationClassifier.py --architecture=custom --seeds_ensemble=10-20 --ensemble=True --batch_size=8
--epochs=10 --routine=cv --folds=5"
echo "$cmd"
eval "$cmd"