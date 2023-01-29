#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

for DOWN_SAMPLE_EF in "True" "False"; do
  for CELL in "LSTM" "GRU"; do
    cmd="python ../RelationClassifier.py --architecture=custom --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
      --down_sample_ef=$DOWN_SAMPLE_EF --epochs=10 --routine=cv --folds=5 --filters=32 --kernel_size=3 --pool_size=2 \
      --rnn_cell=$CELL --rnn_units=32"
    echo "$cmd"
    eval "$cmd"
  done
done