#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# conda activate thesis

# including synonyms (all data; only gateways) with ensembles and best combination of label/weight
for SAMPLING in "normal" "og"; do
    cmd="python ../../GatewayTokenClassifier.py --seeds_ensemble=0-29 --batch_size=8 --epochs=1 --ensemble=True --routine=cv --folds=5 \
        --labels=all --other_labels_weight=0.1 --sampling_strategy=$SAMPLING --use_synonyms=True"
    echo "$cmd"
    eval "$cmd"
done