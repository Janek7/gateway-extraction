#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

# conda activate thesis

cmd="python ../../GatewayTokenClassifier.py --seeds_ensemble=0-29 --store_weights=True --batch_size=8 --epochs=1 --ensemble=True \
          --routine=ft --labels=all --other_labels_weight=0.1 --sampling_strategy=og"
echo "$cmd"
eval "$cmd"