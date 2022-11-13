#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

for LABELS in "all" "filtered"; do
    for WEIGHT in 0.1 0.2 0.3 0.4 0.5 0.75 1.0; do
      cmd="python ../GatewayTokenClassifier_train.py --batch_size=8 --epochs=3 --ensemble=True --routine=cv --folds=5 --routine=cv \
          --labels=$LABELS --other_labels_weight=$WEIGHT --sampling_strategy=normal"
      echo $cmd
      eval $cmd
    done
done

exit 1

# try all sampling methods with ensembles and best combination of label/weight
for SAMPLING in "normal" "up" "down" "og"; do
    cmd="python ../GatewayTokenClassifier_train.py --batch_size=8 --epochs=3 --ensemble=True --routine=cv --folds=5 --routine=cv \
        --labels=all --other_labels_weight=0.1 --sampling_strategy=$SAMPLING"
    echo $cmd
    eval $cmd
done

# including synonyms (all data; only gateways) with ensembles and best combination of label/weight
for SAMPLING in "normal" "og"; do
    cmd="python ../GatewayTokenClassifier_train.py --batch_size=8 --epochs=3 --ensemble=True --routine=cv --folds=5 --routine=cv \
        --labels=all --other_labels_weight=0.1 --sampling_strategy=$SAMPLING --use_synonyms=True"
    echo $cmd
    eval $cmd
done