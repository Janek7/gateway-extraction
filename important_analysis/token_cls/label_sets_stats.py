# add parent dir to sys path for import of modules
import os
import sys
# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

from collections import Counter

from token_approaches.token_data_preparation import preprocess_tokenization_data
from labels import *


def count_labels(label_set):
    _, labels, _, _ = preprocess_tokenization_data(use_synonyms=False,
                                                   sampling_strategy=NORMAL,
                                                   other_labels_weight=.1,
                                                   label_set=label_set,
                                                   activity_masking=NOT)
    label_array = labels.numpy()
    flattened = label_array.flatten()
    # remove padding tokens
    non_zeros = flattened[flattened != 0]
    print(len(non_zeros))
    print(Counter(non_zeros))


count_labels("all")
count_labels("filtered")
