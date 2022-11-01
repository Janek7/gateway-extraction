#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# conda activate thesis

python ../GatewayTokenClassifier.py --seed=42 --batch_size=8 --epochs=3 --routine=single --dev_share=0.1 --labels=filtered --other_labels_weight=0.1
exit 1

for LABEL in "all" "filtered"; do
    for OTHER_LABEL_WEIGHT in 0.1 0.2 0.3 0.4 0.5 0.75 1; do
        echo "python ../GatewayTokenClassifier.py --seed=42 --batch_size=8 --epochs=3 --routine=cv --folds=5 --labels=$LABEL --other_labels_weight=$OTHER_LABEL_WEIGHT"
        python ../GatewayTokenClassifier.py --seed=42 --batch_size=8 --epochs=3 --routine=cv --folds=5 --labels=$LABEL --other_labels_weight=$OTHER_LABEL_WEIGHT
    done
done
