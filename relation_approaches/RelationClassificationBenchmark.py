#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

from typing import Dict, List, Tuple, Callable
import logging
import argparse

import tensorflow as tf
import pandas as pd

from relation_approaches.RelationClassifier import RelationClassifier, NeuralRelationClassifierEnsemble
from relation_approaches.activity_relation_dataset_preparation import create_activity_relation_cls_dataset_full,\
    _create_dataset
from relation_approaches import metrics
from utils import ROOT_DIR, config
from labels import *

logger = logging.getLogger('Relation Classification Benchmark')


# IN PROGRESS
class RelationClassificationBenchmarkNew:
    N_ALL = 1000

    def __init__(self, approach_name: str, relation_classifier: RelationClassifier = None, n=None,
                 output_folder: str = None, round_digits: int = 2) -> None:

        # prepare output folder
        if not output_folder:
            output_folder = os.path.join(ROOT_DIR,
                                         f"data/results_relation_approaches/relation_classification/{approach_name}")
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)


# READY TO USE

ROUND_DIGITS = 2


def evaluate_ensemble_native(approach_name: str, ensemble_path: str) -> None:
    """
    run native evaluation on test set by using one model of the loaded ensemble
    evaluate on label and n-distance level
    :param approach_name: approach_name
    :param ensemble_path: path
    :return:
    """
    logger.info(f"Run evaluation native on each of the single models")
    # 1a) load data
    train, test_dataset, test_relations = create_activity_relation_cls_dataset_full(get_static_args(batch_size=None))
    # 1b) load model
    ensemble = NeuralRelationClassifierEnsemble(ensemble_path=ensemble_path, args=get_static_args(),
                                                train_size=len(train), seeds=[10])
    model = ensemble.models[0]

    # 2) helper method
    def filter_relations(dataset: tf.data.Dataset, relations: List, filter_function: Callable[[Dict], bool]) \
            -> Tuple[tf.data.Dataset, List[Dict]]:
        """
        filter data (tensorflow dataset and relation list) for samples that match the filter function
        :param filter_function: function for evaluating a relation -> Input: relation as dict; Output: bool
        """
        # 1) search for relevant indices
        label_filtered_indexes = [i for i, relation in enumerate(test_relations) if filter_function(relation)]
        # 2a) filter relations for indices
        relations_filtered = [r for i, r in enumerate(relations) if i in label_filtered_indexes]

        # 2b) filter dataset for indices
        filtered_input_ids = []
        filtered_attention_masks = []
        filtered_labels = []
        for i, (x, y) in enumerate(dataset.as_numpy_iterator()):
            if i in label_filtered_indexes:
                filtered_input_ids.append(x["input_ids"])
                filtered_attention_masks.append(x["attention_mask"])
                filtered_labels.append(y)
        filtered_dataset = _create_dataset(tf.constant(filtered_input_ids), tf.constant(filtered_attention_masks),
                                           tf.constant(filtered_labels))
        return filtered_dataset, relations_filtered

    # 3) evaluations
    evaluation_entries = []
    Ns = [1, 2, 5, 10, 30]

    # 3a) evaluate whole test set
    _, _, precision, recall = model.evaluate(test_dataset.batch(8))
    evaluation_entries.append({"label": "all", "n": "all", "precision": precision, "recall": recall,
                               "f1": metrics.f1(precision, recall), "support": len(test_relations)})

    # 3b) create evaluations for filtered relation sets
    for label in [DIRECTLY_FOLLOWING, EVENTUALLY_FOLLOWING, EXCLUSIVE, CONCURRENT]:
        # 3b1) evaluate whole label set
        print(f" Evaluate {label} ... ".center(100, '+'))
        test_dataset_label_filtered, test_relations_label_filtered \
            = filter_relations(test_dataset, test_relations, lambda r: r[RELATION_TYPE] == label)
        _, _, precision, recall = model.evaluate(test_dataset_label_filtered.batch(8))
        evaluation_entries.append({"label": label, "n": "all", "precision": round(precision, ROUND_DIGITS),
                                   "recall": round(recall, ROUND_DIGITS), "f1": round(metrics.f1(precision, recall)),
                                   "support": len(test_relations_label_filtered)})

        # 3b2) evaluate label set splitted in n relations with activity order distance <= n
        for n in Ns:
            print(f" ... {label} && distance <={n} ... ".center(100, '+'))
            test_dataset_label_n_filtered, test_relations_label_n_filtered \
                = filter_relations(test_dataset_label_filtered, test_relations_label_filtered,
                                   lambda r: abs(r[ACTIVITY_1][0] - r[ACTIVITY_2][0]) <= n)
            try:
                _, _, precision, recall = model.evaluate(test_dataset_label_n_filtered.batch(8))
                evaluation_entries.append({"label": label, "n": n, "precision": round(precision, ROUND_DIGITS),
                                           "recall": round(recall, ROUND_DIGITS),
                                           "f1": round(metrics.f1(precision, recall)),
                                           "support": len(test_relations_label_n_filtered)})
            except OverflowError as e:
                evaluation_entries.append({"label": label, "n": n, "precision": 0, "recall": 0, "f1": 0,
                                           "support": len(test_relations_label_n_filtered),
                                           "comment": "error (support == 0?)"})


    # 4) Write results
    evaluation_df = pd.DataFrame.from_dict(evaluation_entries)
    path = os.path.join(ROOT_DIR, "data/results_relation_approaches/relation_classification", approach_name)
    os.makedirs(path, exist_ok=True)
    with pd.ExcelWriter(os.path.join(path, "results.xlsx")) as writer:
        evaluation_df.to_excel(writer, sheet_name='Evaluation', index=False)


def get_static_args(batch_size: int = 8):
    """
    necessary to pass arguments to ensemble and 'create_activity_relation_cls_dataset_full' call in evaluate_ensemble
    IMPORTANT: argument values must match with the ones that were used during training of the ensemble
    :param batch_size: batch_size
    :return:
    """
    parser = argparse.ArgumentParser()
    # Standard params
    parser.add_argument("--batch_size", default=batch_size, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Epochs")
    parser.add_argument("--seed_general", default=42, type=int, help="Random seed.")
    parser.add_argument("--test_docs", default=True, type=bool,
                        help="Flag if predefined docs should be used as test set")
    parser.add_argument("--test_share", default=0.1, type=float, help="Share of test set")
    parser.add_argument("--down_sample_ef", default=False, type=bool,
                        help="Flag if eventually following samples should be"
                             "down sampled to comparable number")
    # Architecture params
    parser.add_argument("--architecture", default=ARCHITECTURE_BRCNN, type=str, help="Architecture variants")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--hidden_layer", default=32, type=int, help="Hidden layer size")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument("--warmup", default=0, type=int, help="Number of warmup steps.")
    # cnn params
    parser.add_argument("--cnn_blocks", default=1, type=int, help="Number of filters in CNN")
    parser.add_argument("--filter_start_size", default=32, type=int,
                        help="Start (minimal) number of filters in first cnn block")
    parser.add_argument("--filter_increase", default=2, type=int,
                        help="Rate how much the number of filters should grow in "
                             "each new block")
    parser.add_argument("--kernel_size", default=3, type=int, help="Kernel size in CNN")
    parser.add_argument("--pool_size", default=2, type=int, help="Max pooling size")
    # rnn params
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="Type of RNN cell (LSTM or GRU)")
    parser.add_argument("--rnn_units", default=128, type=int, help="Number of units in RNNs")
    parser.add_argument("--rnn_backwards", default=False, type=bool,
                        help="Flag if backwards should be processed as well.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    evaluate_ensemble_native("brcnn_128", config[MODELS][ACTIVITY_RELATION_CLASSIFIER])
