#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

for LABELS in "all" "filtered"; do
    for WEIGHT in 0.1 0.2 0.3 0.4 0.5 0.75 1.0; do
      cmd="python ../GatewayTokenClassifier_train.py --seeds_ensemble=0-29 --batch_size=8 --epochs=1 --ensemble=True --routine=cv --folds=5 --routine=cv \
          --labels=$LABELS --other_labels_weight=$WEIGHT --sampling_strategy=normal"
      echo "$cmd"
      eval "$cmd"
    done
done
