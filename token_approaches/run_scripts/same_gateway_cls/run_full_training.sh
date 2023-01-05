#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

# conda activate thesis


cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --store_weights=True --ensemble=True --batch_size=8 --epochs=10 \
      --routine=ft --use_synonyms=True --mode=context_text_and_labels_n_gram --context_size=1 --n_gram=0"
echo "$cmd"
eval "$cmd"
