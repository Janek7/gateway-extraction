#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

# conda activate thesis


cmd="python ../RelationClassifier.py --architecture=brcnn --seeds_ensemble=10-20 --ensemble=True --store_weights=True --batch_size=8 \
      --epochs=10 --routine=ft --dropout=0 --cnn_blocks=1 --filter_start_size=32 \
      --rnn_cell=LSTM --rnn_units=128"
echo "$cmd"
eval "$cmd"
