#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

# try all masking variants
for MASKING in "not" "dummy" "single_mask" "multi_mask"; do
    cmd="python ../../GatewayTokenClassifier_train.py --seeds_ensemble=0-29 --batch_size=8 --epochs=1 --ensemble=True --routine=cv --folds=5 --routine=cv \
        --labels=all --other_labels_weight=0.1 --activity_usage=$MASKING"
    echo "$cmd"
    eval "$cmd"
done
