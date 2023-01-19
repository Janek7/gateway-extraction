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

logger = logging.getLogger('Relation Classification Benchmark')


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
        os.makedirs(self.output_folder, exist_ok=True)
        self.round_digits = round_digits
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
        logger.info(f"Evaluate relations from {len(doc_names)} documents")

        # 1) Create predictions
        relation_predictions = classify_documents(self.relation_classifier, doc_names)

        # 2) Compute metrics per class and document
        all_doc_metrics = self.compute_metrics(relation_predictions, doc_names)

        # 4a) Compute average statistics for whole document set
        label_avg_metrics = self.average_label_wise(all_doc_metrics)

        # 4b) Compute average statistics for whole set by averaging (already averaged) label metrics
        logger.info("Compute average overall metrics")
        overall_avg_metrics = metrics.average_metrics([m for label, m in label_avg_metrics.items()],
                                                      self.round_digits)

        # 5) Write results & predictions into output folder
        self.write_metrics(all_doc_metrics, label_avg_metrics, overall_avg_metrics)
        self.write_results(relation_predictions)

    def compute_metrics(self, relation_predictions, doc_names: List[str]) -> Dict:
        """
        Compute metrics per class and document
        :param relation_predictions: dictionary of relations per document
        :param doc_names: target document names
        :return: dictionary with structure {doc-name: {label: {metric: value}}}
        """
        logger.info("Compute metrics doc-level metrics")
        all_doc_metrics = {}
        for doc_name in doc_names:
            doc_relation_predictions = {doc_name: relation_predictions[doc_name]}
            doc_metrics = {}
            for label in LABELS:
                doc_label_metrics = self.evaluate_activity_relations(doc_relation_predictions, label=label)
                doc_metrics[label] = doc_label_metrics[doc_name]
            all_doc_metrics[doc_name] = doc_metrics
        return all_doc_metrics

    def evaluate_activity_relations(self, relations: Dict, label: str = None) -> Dict:
        """
        evaluate relations against "gold" relations created by activity relation data generation algorithm
        evaluation can be limited to one label by setting label
        internal relations tuple format: (a1, a2, relation_label)
        :param relations: relations given in dictionary (key = doc_name, value: list of activity
                          relations format -> (doc_name, a1, a2, relation_label, comment)
                          HINT: usually just one doc with its relations is passed
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

    def average_label_wise(self, all_doc_metrics: Dict) -> Dict:
        """
        create averaged metrics for each label
        :param all_doc_metrics: metrics for each label in each document (nested dictionary)
        :return: dictionary with labels as key and averaged metric dict
        """
        logger.info("Compute average metrics per label")
        label_avg_metrics = {}
        for label in LABELS:
            label_doc_metrics = [doc_metrics[label] for doc_name, doc_metrics in all_doc_metrics.items()]
            label_avg = metrics.average_metrics(label_doc_metrics, self.round_digits)
            label_avg_metrics[label] = label_avg
        return label_avg_metrics

    def write_metrics(self, all_doc_metrics: Dict, label_avg_metrics: Dict, overall_avg_metrics: Dict) -> None:
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
        path = os.path.join(self.output_folder, 'results.xlsx')
        logger.info(f"Write results to {path}")
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
                file.write("\n"*3)


if __name__ == '__main__':
    b = RelationClassificationBenchmark("baseline_df", RandomBaselineRelationClassifier())
    b.evaluate_documents(["doc-1.1", "doc-1.2"])  # , "doc-1.2"])
