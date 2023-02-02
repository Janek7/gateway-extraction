# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

from abc import ABC, abstractmethod
from typing import Dict, List

from petreader.labels import *
import pandas as pd

from relation_approaches import metrics
from utils import flatten_list


class AbstractClassificationBenchmark(ABC):
    """
    abstract base class for benchmark classes that evaluate classification scenarios on PET dataset
    """
    def __init__(self, labels: List[str], approach_name: str, output_folder: str, round_digits: int,
                 support_weighted_average: bool = True):
        """
        init a classification benchmark
        :param labels: list of labels (e.g. to use in average label wise)
                       should be set static in constructor of subclass to set of labels of the respective cls problem
                            AND NOT for every evaluation different
        :param approach_name: approach name of the benchmark
        :param output_folder: output folder to store results/predictions
        :param round_digits: number of digits to round metrics to
        :param support_weighted_average: flag if metrics averaging should be weighted by the respective support values
        """
        self.labels = labels
        self.approach_name = approach_name
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.round_digits = round_digits
        self.support_weighted_average = support_weighted_average

    def compute_metrics_dict(self, tp: int, fp: int, fn: int, gold_length: int) -> Dict:
        precision = metrics.precision(tp, fp)
        recall = metrics.recall(tp, fn)
        f1 = metrics.f1(precision, recall)

        doc_metrics = {
            TRUE_POSITIVE: tp,
            FALSE_POSITIVE: fp,
            FALSE_NEGATIVE: fn,
            PRECISION: round(precision, self.round_digits),
            RECALL: round(recall, self.round_digits),
            F1SCORE: round(f1, self.round_digits),
            SUPPORT: gold_length
        }
        return doc_metrics

    def average_label_wise(self, all_doc_metrics: Dict) -> Dict:
        """
        create averaged metrics for each label
        :param all_doc_metrics: metrics for each label in each document (nested dictionary)
        :return: dictionary with labels as key and averaged metric dict
        """
        label_avg_metrics = {}
        for label in self.labels:
            label_doc_metrics = [doc_metrics[label] for doc_name, doc_metrics in all_doc_metrics.items()]
            label_avg = metrics.average_metrics(label_doc_metrics, self.round_digits, self.support_weighted_average)
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
