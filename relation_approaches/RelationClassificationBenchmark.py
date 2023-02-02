#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

from typing import Dict, List, Tuple
from copy import deepcopy
import logging
import json
import argparse

from relation_approaches.AbstractClassificationBenchmark import AbstractClassificationBenchmark
from relation_approaches.activity_relation_data_preparation import get_activity_relations, DOC_BLACK_LIST
from relation_approaches.RelationClassifier import RelationClassifier, classify_documents, \
    NeuralRelationClassifierEnsemble, create_relation_benchmark_format, DFBaselineRelationClassifier, \
    RandomBaselineRelationClassifier
from relation_approaches.activity_relation_dataset_preparation import label_dict, \
    create_activity_relation_cls_dataset_full, TEST_DOCS
from relation_approaches import metrics
from utils import ROOT_DIR, GatewayExtractionException
from PetReader import pet_reader
from labels import *

logger = logging.getLogger('Relation Classification Benchmark')


class RelationClassificationBenchmark(AbstractClassificationBenchmark):
    """
    Creates and evaluates predictions of activity relation pairs using a given RelationClassifier instance
    """
    # parameter for n value to evaluate all activity pairs
    N_ALL = 1000

    def __init__(self, approach_name: str, relation_classifier: RelationClassifier = None, n=None,
                 output_folder: str = None, round_digits: int = 2) -> None:
        """
        initialize a RelationClassificationBenchmark
        :param approach_name: approach name which defines results folder name
        :param relation_classifier: RelationClassifier instance to evaluate
        :param n: limit of evaluations of metrics of the next n activites of each activity (int or list)
        :param output_folder: output folder; if None, generate based on approach_approach_name
        :param round_digits: number of digits to round metrics
        """
        self.relation_classifier = relation_classifier
        # define nearest n's to check & append a arbitrary large number to evaluate all
        if n is None:
            self.n = [1, 2, 5, 10]
        elif isinstance(n, list):
            self.n = n
        elif isinstance(n, int):
            self.n = list(range(1, n + 1))
        self.n.append(self.N_ALL)

        # load gold standard
        self.gold_activity_relations = get_activity_relations()

        # prepare output folder
        if not output_folder:
            output_folder = os.path.join(ROOT_DIR,
                                         f"data/results_relation_approaches/relation_classification/{approach_name}")

        AbstractClassificationBenchmark.__init__(self, list(label_dict.keys()), approach_name, output_folder,
                                                 round_digits)

    def evaluate_documents(self, doc_names: List[str], relation_predictions: Dict[str, List] = None):
        """
        evaluate list of documents with relation_classifier
        :param doc_names: doc names, if none -> all
        :param relation_predictions: already ready to use predictions organized in dictionary per document
                                     relations in list only consist of (activity1, activity2, relation_type) tuples
                                     if None -> create with self.relation_classifier and classify_documents
        :return:
        """
        if not doc_names:
            doc_names = pet_reader.document_names
            doc_names = [d for d in doc_names if d not in DOC_BLACK_LIST]
        if not relation_predictions:
            logger.info(f"Create predictions for {len(doc_names)} documents")
            relation_predictions = classify_documents(self.relation_classifier, doc_names)

        logger.info(f"Evaluate predictions with set of n's: {self.n}")
        all_doc_metrics_nearest_n = self.compute_document_label_metrics_nearest_n(relation_predictions)
        label_avg_metrics_nearest_n = {i: self.average_label_wise(m) for i, m in all_doc_metrics_nearest_n.items()}
        overall_avg_metrics_n = {i: metrics.average_metrics([m for label, m in lm.items()], self.round_digits)
                                 for i, lm in label_avg_metrics_nearest_n.items()}

        logger.info(f"Write results & predictions to {self.output_folder}")
        self.write_results(relation_predictions)
        for i in self.n:
            name = "all" if i == self.N_ALL else f"n={str(i)}"
            self.write_metrics(all_doc_metrics_nearest_n[i], label_avg_metrics_nearest_n[i], overall_avg_metrics_n[i],
                               name=name)

    def compute_document_label_metrics(self, relation_predictions: Dict[str, List],
                                       gold_relations: Dict[str, List] = None) -> Dict[str, Dict]:
        """
        Compute metrics per class and document
        :param relation_predictions: dictionary of predicted relations per document
        :param gold_relations: dictionary of gold relations per document
        :return: dictionary with structure {doc-name: {label: {metric: value}}}
        """
        all_doc_metrics = {}
        for doc_name in relation_predictions.keys():
            doc_metrics = {}
            for label in self.labels:
                doc_label_metrics = self.evaluate_activity_relations(doc_name, relation_predictions[doc_name],
                                                                     gold_relations=gold_relations[doc_name], label=label)
                doc_metrics[label] = doc_label_metrics
            all_doc_metrics[doc_name] = doc_metrics
        return all_doc_metrics

    def compute_document_label_metrics_nearest_n(self, relation_predictions: Dict[str, List]) -> Dict[int, Dict]:
        """
        compute document/label metrics n times with limiting relations to evaluate to activities that are within a
        distance of n in the sequence of activities in the whole document
        :param relation_predictions: dictionary of relations per document
        :return: dict with n as key and dictionary with structure {doc-name: {label: {metric: value}}} as value
        """
        all_doc_metrics_nearest_n = {}
        for i in self.n:

            # limit relation_predictions and gold_relations to activity pairs within range of n
            gold_relations_limited = {}
            relation_predictions_limited = {}

            for doc_name, doc_relations in relation_predictions.items():
                # extract gold relations of document and limit to (a1, a2, relation_type)
                doc_gold_relations = [r[1:4] for r in self.gold_activity_relations if r[0] == doc_name]

                # limit relation_predictions and gold_relations to activity pairs within range of n
                # add to new result set
                relation_predictions_limited[doc_name] = self.limit_relations_to_nearest_n(doc_name, doc_relations, i)
                gold_relations_limited[doc_name] = self.limit_relations_to_nearest_n(doc_name, doc_gold_relations, i)

            # create predictions for limited sets as in normal version
            preds = self.compute_document_label_metrics(relation_predictions_limited, gold_relations_limited)
            all_doc_metrics_nearest_n[i] = preds
        return all_doc_metrics_nearest_n

    @staticmethod
    def limit_relations_to_nearest_n(doc_name: str, relations: List[Tuple], n) -> List[Tuple]:
        """
        limit relations pairs to the ones with a order distance between activities <= n
        :param doc_name: doc name
        :param relations: list of relations to filter
        :param n: maximum distance threshold
        :return: filtered list
        """
        activity_order = pet_reader.get_activities_in_relation_approach_format(doc_name)
        return [r for r in relations if abs(activity_order.index(r[0]) - activity_order.index(r[1])) <= n]

    def evaluate_activity_relations(self, doc_name: str, relations: List, gold_relations: List = None,
                                    label: str = None) -> Dict:
        """
        evaluate relations of a document against "gold" relations created by activity relation data generation algorithm
        evaluation can be limited to one label by setting label
        :param doc_name: doc name
        :param relations: relations given in dictionary (key = doc_name, value: list of activity
                          relations format -> (doc_name, a1, a2, relation_label, comment))
                          HINT: usually just one doc with its relations is passed
        :param gold_relations: gold relations in same format; if none -> take from normal gold standard
                               (can be passed for evaluating nearest n with method compute_document_label_metrics_nearest_n)
        :param label: if set, only one class is evaluated
        :return: dictionary with metrics for each document
        """
        if not gold_relations:
            # filter gold_relation on document and reduce relation to format (a1, a2, label)
            gold_relations = [r[1:4] for r in self.gold_activity_relations if r[0] == doc_name]

        if len(relations) != len(gold_relations):
            msg = f"Predictions and gold standard of {doc_name} have not the same length " \
                  f"(label={label}) ({len(relations)} vs. {len(gold_relations)})"
            # raise GatewayExtractionException(msg)
            logger.warning(msg)

        pred_relations = deepcopy(relations)
        gold_relations = deepcopy(gold_relations)

        if label:
            pred_relations = [r for r in pred_relations if r[2] == label]
            gold_relations = [r for r in gold_relations if r[2] == label]

        # define counters
        tp = 0
        fp = 0
        fn = 0
        # tn does not exist in this use case

        for gold_relation in gold_relations:
            for pred_relation in pred_relations:
                # check both activity orders
                if gold_relation == pred_relation \
                        or gold_relation == (pred_relation[1], pred_relation[0], pred_relation[2]):
                    tp += 1
                    pred_relations.remove(pred_relation)
                    break
        # fp = number of elements in predictions that remain unmatched
        fp = len(pred_relations)
        # fn = number of elements in gold standard that remain unmatched
        fn = len(gold_relations) - tp

        # store in metrics dict
        doc_metrics = self.compute_metrics_dict(tp, fp, fn, len(gold_relations))
        return doc_metrics

    def write_results(self, relation_predictions: Dict) -> None:
        """
        write single predictions to json and txt file
        :param relation_predictions: dictionary of relations predictions document-wise
        :return:
        """
        with open(os.path.join(self.output_folder, "predictions.json"), 'w') as file:
            json.dump(relation_predictions, file, indent=4)

        with open(os.path.join(self.output_folder, "predictions.txt"), 'w') as file:
            for doc_name, relations in relation_predictions.items():
                file.write(f" {doc_name} ".center(100, '-') + "\n")
                for r in relations:
                    file.write(str(r) + "\n")
                file.write("\n" * 3)


def evaluate_ensemble(approach_name: str, ensemble_path: str):
    """
    evaluate a ensemble stored in given ensemble path
    :param approach_name: approach name
    :param ensemble_path: path
    :return: writes to folder
    """
    if not ensemble_path:
        # load ensemble ensemble
        ensemble = NeuralRelationClassifierEnsemble(seeds=[3, 4], args=get_dummy_args())
    else:
        ensemble = NeuralRelationClassifierEnsemble(ensemble_path=ensemble_path)

    # load data and predict test set
    _, test, test_relations = create_activity_relation_cls_dataset_full(get_dummy_args())
    test_predictions = ensemble.predict_test_set(test)
    test_relations_predicted = create_relation_benchmark_format(test_predictions, test_relations)

    # evaluate with benchmark by passing test doc list and relations to evaluate_documents
    b = RelationClassificationBenchmark(approach_name)
    b.evaluate_documents(TEST_DOCS, test_relations_predicted)


def get_dummy_args():
    """
    necessary to pass arguments to ensemble and 'create_activity_relation_cls_dataset_full' call in evaluate_ensemble
    IMPORTANT: argument values must match with the ones that were used during training of the ensemble
    :return:
    """
    parser = argparse.ArgumentParser()
    # Standard params
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--seed_general", default=42, type=int, help="Random seed.")
    parser.add_argument("--test_docs", default=True, type=bool,
                        help="Flag if predefined docs should be used as test set")
    parser.add_argument("--test_share", default=0.1, type=float, help="Share of test set")
    parser.add_argument("--down_sample_ef", default=False, type=bool,
                        help="Flag if eventually following samples should be"
                             "down sampled to comparable number")
    # Architecture params
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

    args = parser.parse_args([] if "__file__" not in globals() else None)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    return args


if __name__ == '__main__':
    #b = RelationClassificationBenchmark("baseline_random", RandomBaselineRelationClassifier())
    #b.evaluate_documents(TEST_DOCS)

    # evaluate_ensemble("custom_random", ensemble_path=None)
    evaluate_ensemble("brcnn_128", ensemble_path="/home/japutz/master-thesis/data/final_models/RelationClassifier-2023-02-02_104323-a=brcnn,bs=8,cb=1,dse=False,d=0.0,e=True,e=10,fi=2,fss=32,f=5,hl=32,ks=3,lr=2e-05,ps=2,rb=False,rc=LSTM,ru=128,r=ft,sg=42,se=10-20,sw=True,td=True,ts=0.1,w=0")
