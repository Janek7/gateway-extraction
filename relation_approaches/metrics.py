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


def average_metrics(metric_list: List[Dict], round_digits=2) -> Dict:
    """
    Average list of metrics (except SUPPORT -> sum)
    :param metric_list: list of metrics dicts
    :param round_digits: number of digits to round the values to
    :return: metric dict with averaged values
    """
    metric_keys = [PRECISION, RECALL, F1SCORE, SUPPORT]
    averaged_metrics = {m: round(float(np.mean([doc_metrics[m] for doc_metrics in metric_list])), round_digits)
                        if m != SUPPORT else sum([doc_metrics[m] for doc_metrics in metric_list])
                        for m in metric_keys}
    return averaged_metrics


if __name__ == '__main__':
    data = [
        [0.8816,    0.8394],
        [0.7674,    0.666],
        [0.8773,    0.8357],
        [0.7526,    0.6536],
    ]
    for p, r in data:
        print(round(f1(p, r), 4))

