#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

# conda activate thesis


# 1) try lstm/gru and down sampling/no sampling

# note: down_sample_ef is a boolean argument -> only existence (even if set to False) is interpreted as True
# for DOWN_SAMPLE_EF in "--down_sample_ef=True" ""; do
#for DOWN_SAMPLE_EF in ""; do
#  for CELL in "LSTM" "GRU"; do
#    cmd="python ../RelationClassifier.py --architecture=brcnn --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
#      $DOWN_SAMPLE_EF --epochs=10 --routine=cv --folds=5 --dropout=0 --cnn_blocks=1 --filter_start_size=32 \
#      --rnn_cell=$CELL --rnn_units=32"
#    echo "$cmd"
#    eval "$cmd"
#  done
#done


# 2) try different CNN stackings

#for NUMBER_BLOCKS in "2" "3" "4"; do
#    cmd="python ../RelationClassifier.py --architecture=brcnn --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
#      --epochs=10 --routine=cv --folds=5 --dropout=0 --cnn_blocks=$NUMBER_BLOCKS --filter_start_size=32 \
#      --rnn_cell=LSTM --rnn_units=32"
#    echo "$cmd"
#    eval "$cmd"
#  done


# 3) try different RNN cell sizes

for UNITS in "64" "128"; do
    cmd="python ../RelationClassifier.py --architecture=brcnn --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
      --epochs=10 --routine=cv --folds=5 --dropout=0 --cnn_blocks=1 --filter_start_size=32 \
      --rnn_cell=LSTM --rnn_units=$UNITS"
    echo "$cmd"
    eval "$cmd"
done


# 4) try adding dropout

for UNITS in "32" "64" "128"; do
  cmd="python ../RelationClassifier.py --architecture=brcnn --seeds_ensemble=10-20 --ensemble=True --batch_size=8 \
    --epochs=10 --routine=cv --folds=5 --dropout=0.2 --cnn_blocks=1 --filter_start_size=32 \
    --rnn_cell=LSTM --rnn_units=$UNITS"
  echo "$cmd"
  eval "$cmd"
done
