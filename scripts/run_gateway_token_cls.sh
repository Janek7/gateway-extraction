#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

python ../GatewayTokenClassifier_train.py --batch_size=8 --epochs=3 --ensemble=True --routine=cv --folds=5 --routine=cv \
            --labels=all --other_labels_weight=0.1 --sampling_strategy=normal
exit 1

# try all sampling methods with ensembles and best combination of label/weight
for SAMPLING in "normal" "up" "down" "og"; do
    python ../GatewayTokenClassifier_train.py --batch_size=8 --epochs=3 --ensemble=True --routine=cv --folds=5 --routine=cv \
        --labels=all --other_labels_weight=0.1 --sampling_strategy=$SAMPLING
done

# including synonyms (all data; only gateways) with ensembles and best combination of label/weight
for SAMPLING in "normal" "og"; do
    python ../GatewayTokenClassifier_train.py --batch_size=8 --epochs=3 --ensemble=True --routine=cv --folds=5 --routine=cv \
        --labels=all --other_labels_weight=0.1 --sampling_strategy=$SAMPLING --use_synonyms=True
done