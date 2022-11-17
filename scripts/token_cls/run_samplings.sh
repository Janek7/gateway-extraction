#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

# try all sampling methods with ensembles and best combination of label/weight
for SAMPLING in "normal" "up" "down" "og"; do
    cmd="python ../GatewayTokenClassifier_train.py --seeds_ensemble=0-29 --batch_size=8 --epochs=1 --ensemble=True --routine=cv --folds=5 --routine=cv \
        --labels=all --other_labels_weight=0.1 --sampling_strategy=$SAMPLING"
    echo "$cmd"
    eval "$cmd"
done
