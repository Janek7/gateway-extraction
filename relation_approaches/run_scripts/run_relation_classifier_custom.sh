#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# conda activate thesis

# note: down_sample_ef is a boolean argument -> only existence (even if set to False) is interpreted as True
# for DOWN_SAMPLE_EF in "--down_sample_ef=True" ""; do
for DOWN_SAMPLE_EF in ""; do
  cmd="python ../RelationClassifier.py --architecture=custom --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
    $DOWN_SAMPLE_EF --epochs=10 --routine=cv --folds=5"
  echo "$cmd"
  eval "$cmd"
done