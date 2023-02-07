from typing import List, Dict

import numpy as np
from petreader.labels import *

EPSILON = 1e-10


def precision(tp, fp):
    return tp / (tp + fp + EPSILON)


def recall(tp, fn):
    return tp / (tp + fn + EPSILON)


def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + EPSILON)


def average_metrics(metric_list: List[Dict], round_digits: int = 2, support_weighted_average: bool = True) -> Dict:
    """
    Average list of metrics (except SUPPORT -> sum)
    metric values are averaged by SUPPORT values of the respective label
    :param metric_list: list of metrics dicts
    :param round_digits: number of digits to round the values to
    :param support_weighted_average: flag if metrics averaging should be weighted by the respective support values
    :return: metric dict with averaged values
    """
    metric_keys = [PRECISION, RECALL, F1SCORE, SUPPORT]
    if support_weighted_average:
        averaged_metrics = {m: round(float(sum([doc_metrics[m] * doc_metrics[SUPPORT] for doc_metrics in metric_list]) /
                                           sum([doc_metrics[SUPPORT] for doc_metrics in metric_list])
                                           if sum([doc_metrics[SUPPORT] for doc_metrics in metric_list]) else 0),
                                     round_digits)
                            if m != SUPPORT else sum([doc_metrics[m] for doc_metrics in metric_list])
                            for m in metric_keys}
    else:
        averaged_metrics = {m: round(float(np.mean([doc_metrics[m] for doc_metrics in metric_list
                                                    if doc_metrics[SUPPORT] > 0])),
                                     round_digits)
                            if m != SUPPORT else sum([doc_metrics[m] for doc_metrics in metric_list])
                            for m in metric_keys}
    return averaged_metrics


if __name__ == '__main__':
    data = [
        [0.45,	0.62],
    ]
    for p, r in data:
        print(round(f1(p, r), 4))

