#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

# conda activate thesis

# Mode = N-gram
cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --use_synonyms=True --mode=n_gram --n_gram=1"
echo "$cmd"
eval "$cmd"

# Mode = Context & Index
cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --use_synonyms=True --mode=context_index --context_size=0"
echo "$cmd"
eval "$cmd"

# Mode = Context & n_gram
cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --use_synonyms=True --mode=context_n_gram --context_size=1 --n_gram=0"
echo "$cmd"
eval "$cmd"

# Mode = Context token labels & n_gram
cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --use_synonyms=True --mode=context_labels_n_gram --context_size=0 --n_gram=0"
echo "$cmd"
eval "$cmd"

# Mode = Context & Context token labels & n_gram
cmd="python ../../SameGatewayClassifier.py --seeds_ensemble=10-20 --ensemble=True --batch_size=8 --epochs=10 \
      --routine=cv --folds=5 --use_synonyms=True --mode=context_text_and_labels_n_gram --context_size=1 --n_gram=0"
echo "$cmd"
eval "$cmd"
