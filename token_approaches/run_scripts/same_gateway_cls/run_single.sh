#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis


cmd="python ../../SameGatewayClassifier_train.py --seeds_ensemble=0-29 --ensemble=True --batch_size=8 --epochs=3 \
      --routine=cv --folds=5 --routine=cv"
echo "$cmd"
eval "$cmd"
