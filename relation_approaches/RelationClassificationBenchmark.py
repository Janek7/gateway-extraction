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

from petreader.labels import *
import pandas as pd

from relation_approaches.activity_relation_data_preparation import get_activity_relations
from relation_approaches.RelationClassifier import RelationClassifier, GoldstandardRelationClassifier, \
    DFBaselineRelationClassifier, RandomBaselineRelationClassifier, classify_documents
from relation_approaches import metrics
from utils import ROOT_DIR, flatten_list
from PetReader import pet_reader
from labels import *

# activity relation types/labels
LABELS = [DF, EXCLUSIVE, CONCURRENT, NON_RELATED]
# parameter for n value to evaluate all activity pairs
N_ALL = 1000

logger = logging.getLogger('Relation Classification Benchmark')


class RelationClassificationBenchmark:
    """
    Creates and evaluates predictions of activity relation pairs using a given RelationClassifier instance
    """

    def __init__(self, approach_name: str, relation_classifier: RelationClassifier, n=None,
                 output_folder: str = None, round_digits: int = 2) -> None:
        """
        initialize a RelationClassificationBenchmark
        :param approach_name: approach name which defines results folder name
        :param relation_classifier: RelationClassifier instance to evaluate
        :param n: limit of evaluations of metrics of the next n activites of each activity (int or list)
        :param output_folder: output folder; if None, generate based on approach_approach_name
        :param round_digits: number of digits to round metrics
        """
        self.approach_name = approach_name
        self.relation_classifier = relation_classifier
        self.round_digits = round_digits
        # define nearest n's to check & append a arbitrary large number to evaluate all
        if n is None:
            self.n = [1, 2, 5, 10]
        elif isinstance(n, list):
            self.n = n
        elif isinstance(n, int):
            self.n = list(range(1, n + 1))
        n.append(N_ALL)
        # prepare output folder
        if output_folder:
            self.output_folder = output_folder
        else:
            self.output_folder = os.path.join(ROOT_DIR, f"data/results_relation_approaches/relation_classification"
                                                        f"/{approach_name}")
        os.makedirs(self.output_folder, exist_ok=True)
        # load gold standard
        self.gold_activity_relations = get_activity_relations()

    def evaluate_documents(self, doc_names: List[str]):
        """
        evaluate list of documents with relation_classifier
        :param doc_names: doc names, if none -> all
        :return:
        """
        if not doc_names:
            doc_names = pet_reader.document_names
        logger.info(f"Create predictions for {len(doc_names)} documents")
        relation_predictions = classify_documents(self.relation_classifier, doc_names)

        logger.info(f"Evaluate predictions with set of n's: {self.n}")
        all_doc_metrics_nearest_n = self.compute_document_label_metrics_nearest_n(relation_predictions, doc_names)
        label_avg_metrics_nearest_n = {i: self.average_label_wise(m) for i, m in all_doc_metrics_nearest_n.items()}
        overall_avg_metrics_n = {i: metrics.average_metrics([m for label, m in lm.items()], self.round_digits)
                                 for i, lm in label_avg_metrics_nearest_n.items()}

        logger.info(f"Write results & predictions to {self.output_folder}")
        self.write_results(relation_predictions)
        for i in self.n:
            name = "all" if i == N_ALL else f"n={str(i)}"
            self.write_metrics(all_doc_metrics_nearest_n[i], label_avg_metrics_nearest_n[i], overall_avg_metrics_n[i],
                               name=name)

    def compute_document_label_metrics(self, relation_predictions: Dict[str, List], doc_names: List[str],
                                       gold_relations: Dict[str, List] = None) -> Dict[str, Dict]:
        """
        Compute metrics per class and document
        :param relation_predictions: dictionary of predicted relations per document
        :param doc_names: target document names
        :param gold_relations: dictionary of gold relations per document
        :return: dictionary with structure {doc-name: {label: {metric: value}}}
        """
        all_doc_metrics = {}
        for doc_name in doc_names:
            doc_relation_predictions = {doc_name: relation_predictions[doc_name]}
            doc_metrics = {}
            for label in LABELS:
                doc_label_metrics = self.evaluate_activity_relations(doc_relation_predictions,
                                                                     gold_relations=gold_relations, label=label)
                doc_metrics[label] = doc_label_metrics[doc_name]
            all_doc_metrics[doc_name] = doc_metrics
        return all_doc_metrics

    def compute_document_label_metrics_nearest_n(self, relation_predictions, doc_names: List[str]) \
            -> Dict[int, Dict]:
        """
        compute document/label metrics n times with limiting relations to evaluate to activities that are within a
        distance of n in the sequence of activities in the whole document
        :param relation_predictions: dictionary of relations per document
        :param doc_names: target document names
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
            preds = self.compute_document_label_metrics(relation_predictions_limited, doc_names, gold_relations_limited)
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

    def evaluate_activity_relations(self, relations: Dict, gold_relations: Dict = None, label: str = None) -> Dict:
        """
        evaluate relations against "gold" relations created by activity relation data generation algorithm
        evaluation can be limited to one label by setting label
        internal relations tuple format: (a1, a2, relation_label)
        :param relations: relations given in dictionary (key = doc_name, value: list of activity
                          relations format -> (doc_name, a1, a2, relation_label, comment)
                          HINT: usually just one doc with its relations is passed
        :param gold_relations: gold relations in same format; if none -> take from normal gold standard
                               (can be passed for evaluating nearest n with method compute_document_label_metrics_nearest_n)
        :param label: if set, only one class is evaluated
        :return: dictionary with metrics for each document
        """
        if not gold_relations:
            # filter gold_relation on data in scope (doc_names passed in relations) and reduce relation to format (a1, a2, label)
            gold_relations = {doc_name: [r[1:4] for r in self.gold_activity_relations if r[0] == doc_name]
                              for doc_name in relations.keys()}

        doc_metrics = {}
        for doc_name in relations.keys():
            pred_relations = deepcopy(relations[doc_name])
            gold_relations = deepcopy(gold_relations[doc_name])

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

    def average_label_wise(self, all_doc_metrics: Dict) -> Dict:
        """
        create averaged metrics for each label
        :param all_doc_metrics: metrics for each label in each document (nested dictionary)
        :return: dictionary with labels as key and averaged metric dict
        """
        label_avg_metrics = {}
        for label in LABELS:
            label_doc_metrics = [doc_metrics[label] for doc_name, doc_metrics in all_doc_metrics.items()]
            label_avg = metrics.average_metrics(label_doc_metrics, self.round_digits)
            label_avg_metrics[label] = label_avg
        return label_avg_metrics

    def write_metrics(self, all_doc_metrics: Dict, label_avg_metrics: Dict, overall_avg_metrics: Dict,
                      name: str = None) -> None:
        """
        Write all metrics in a normalized version to excel
        """
        # prepare outputs
        all_doc_metrics_list = flatten_list([[{**{"doc_name": doc_name, "label": label}, **one_label_metrics}
                                              for label, one_label_metrics in label_metrics.items()]
                                             for doc_name, label_metrics in all_doc_metrics.items()])
        all_doc_metrics_df = pd.DataFrame.from_dict(all_doc_metrics_list)

        label_avg_metrics_list = [{**{"label": label}, **one_label_metrics}
                                  for label, one_label_metrics in label_avg_metrics.items()]
        label_avg_metrics_df = pd.DataFrame.from_dict(label_avg_metrics_list)

        overall_avg_metrics_df = pd.DataFrame.from_dict([overall_avg_metrics])

        # write to excel
        path = os.path.join(self.output_folder, f'results{f"-{name}" if name else ""}.xlsx')
        with pd.ExcelWriter(path) as writer:
            all_doc_metrics_df.to_excel(writer, sheet_name='Doc-level metrics', index=False)
            label_avg_metrics_df.to_excel(writer, sheet_name='Label-wise metrics', index=False)
            overall_avg_metrics_df.to_excel(writer, sheet_name='Overall metrics', index=False)

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


if __name__ == '__main__':
    b = RelationClassificationBenchmark("baseline_df", DFBaselineRelationClassifier())
    b.evaluate_documents(["doc-1.1", "doc-1.2"])  # , "doc-1.2"])
