# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

from typing import Dict, List
from copy import deepcopy

from petreader.labels import *

from relation_approaches.activity_relation_data_preparation import get_activity_relations
from relation_approaches.RelationClassifier import RelationClassifier, GoldstandardRelationClassifier, \
    DFBaselineRelationClassifier, RandomBaselineRelationClassifier, classify_documents
from relation_approaches import metrics
from utils import ROOT_DIR
from PetReader import pet_reader
from labels import *


# activity relation types/labels
LABELS = [DF, EXCLUSIVE, CONCURRENT, NON_RELATED]


class RelationClassificationBenchmark:

    def __init__(self, approach_name: str, relation_classifier: RelationClassifier,
                 output_folder: str = None, round_digits: int = 2) -> None:
        """
        initialize a RelationClassificationBenchmark
        :param approach_name: approach name which defines results folder name
        :param relation_classifier: RelationClassifier instance to evaluate
        :param output_folder: output folder; if None, generate based on approach_approach_name
        :param round_digits: number of digits to round metrics
        """
        self.approach_name = approach_name
        self.relation_classifier = relation_classifier
        if output_folder:
            self.output_folder = output_folder
        else:
            self.output_folder = os.path.join(ROOT_DIR, f"data/results_relation_approaches/relation_classification"
                                                        f"/{approach_name}")
        self.round_digits = round_digits
        self.gold_activity_relations = get_activity_relations()

    def evaluate_documents(self, doc_names: List[str]):
        """
        evaluate list of documents with relation_classifier
        :param doc_names: doc names, if none -> all
        :return:
        """
        if not doc_names:
            doc_names = pet_reader.document_names

        # 1) Create predictions
        relation_predictions = classify_documents(self.relation_classifier, doc_names)

        # 2) Compute metrics per class and document
        for doc_name in doc_names:
            print(doc_name.center(100, '-'))
            doc_relation_predictions = {doc_name: relation_predictions[doc_name]}
            for label in LABELS:
                doc_label_metrics = self.evaluate_activity_relations(doc_relation_predictions, label=label)
                print(label, doc_label_metrics)

        # 4) Compute average statistics for whole document set
        # averaged_metrics = metrics.average_metrics(list(doc_metrics.values()), self.round_digits)
        # print(averaged_metrics)

        # 3) Write results & statistics into output folder

    def evaluate_activity_relations(self, relations: Dict, label: str = None) -> Dict:
        """
        evaluate relations against "gold" relations created by activity relation data generation algorithm
        evaluation can be limited to one label by setting label
        internal relations tuple format: (a1, a2, relation_label)
        :param relations: relations given in dictionary (key = doc_name, value: list of activity
                          relations format -> (doc_name, a1, a2, relation_label, comment)
        :param label: if set, only one class is evaluated
        :return: dictionary with metrics for each document
        """
        # filter gold_relation on data in scope (doc_names passed in relations) and reduce relation to (a1, a2, label)
        gold_standard = {doc_name: [r[1:4] for r in self.gold_activity_relations if r[0] == doc_name]
                         for doc_name in relations.keys()}

        doc_metrics = {}
        for doc_name in relations.keys():
            pred_relations = deepcopy(relations[doc_name])
            gold_relations = deepcopy(gold_standard[doc_name])

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

            # compute metrics
            precision = metrics.precision(tp, fp)
            recall = metrics.recall(tp, fn)
            f1 = metrics.f1(precision, recall)

            # store in metrics dict
            doc_metrics[doc_name] = {
                TRUE_POSITIVE: tp,
                FALSE_POSITIVE: fp,
                FALSE_NEGATIVE: fn,
                PRECISION: round(precision, self.round_digits),
                RECALL: round(recall, self.round_digits),
                F1SCORE: round(f1, self.round_digits),
                SUPPORT: len(gold_relations)
            }

        return doc_metrics


if __name__ == '__main__':
    b = RelationClassificationBenchmark("baseline_df", DFBaselineRelationClassifier())
    b.evaluate_documents(["doc-1.1", "doc-1.2"])#, "doc-1.2"])
