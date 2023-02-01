#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

# conda activate thesis

# note: down_sample_ef is a boolean argument -> only existence (even if set to False) is interpreted as True
#for DOWN_SAMPLE_EF in "--down_sample_ef=True" ""; do
#  cmd="python ../RelationClassifier.py --architecture=cnn --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
#    $DOWN_SAMPLE_EF --epochs=10 --routine=cv --folds=5 --cnn_blocks=1 --filter_start_size=32"
#  echo "$cmd"
#  eval "$cmd"
#done

for NUMBER_BLOCKS in "1" "2" "3" "4"; do
  cmd="python ../RelationClassifier.py --architecture=cnn --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
    --epochs=10 --routine=cv --folds=5 --cnn_blocks=$NUMBER_BLOCKS --filter_start_size=32"
  echo "$cmd"
  eval "$cmd"
done