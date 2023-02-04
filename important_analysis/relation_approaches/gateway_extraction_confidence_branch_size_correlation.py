# add parent dir to sys path for import of modules
import json
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import json
from typing import List, Tuple

from petreader.labels import *
import numpy as np
import matplotlib.pyplot as plt

from utils import ROOT_DIR


def merge_confidence_size_pairs(confidence_size_pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    merge confidence/size pairs by size to get one averaged point (confidence score) for each size the plot
    """
    unique_sizes = {}
    for size, confidence in confidence_size_pairs:
        if size in unique_sizes:
            unique_sizes[size].append(confidence)
        else:
            unique_sizes[size] = [confidence]
    confidence_size_pairs_merged = [(size, np.mean(confidences)) for size, confidences in unique_sizes.items()]
    confidence_size_pairs_merged.sort(key=lambda pair: pair[0])
    return confidence_size_pairs_merged


def plot_confidence_size_correlation(gateway_extractions_file: str, plot_name: str) -> None:
    """
    creates a graph that plots on y confidence scores and on x gateway sizes (number of activities in branches)
    :param gateway_extractions_file: root dir relative path to json file with extractions
    :param plot_name: short filename
    :return: image output to directory
    """
    # extract data
    with open(os.path.join(ROOT_DIR, gateway_extractions_file)) as file:
        extractions = json.load(file)
    confidence_scores = {XOR_GATEWAY: [], AND_GATEWAY: []}
    for doc_name, gateways in extractions.items():
        for gateway in gateways:
            if gateway["type"] == XOR_GATEWAY:
                confidence_scores[XOR_GATEWAY].append((gateway["size"], gateway["confidence"]))
            elif gateway["type"] == AND_GATEWAY:
                confidence_scores[AND_GATEWAY].append((gateway["size"], gateway["confidence"]))

    # plot
    fig, ax = plt.subplots()
    confidence_scores_merged = {}
    for gateway_type, color in [(XOR_GATEWAY, "cornflowerblue"), (AND_GATEWAY, "mediumseagreen")]:
        confidence_scores_merged[gateway_type] = merge_confidence_size_pairs(confidence_scores[gateway_type])
        ax.plot([pair[0] for pair in confidence_scores_merged[gateway_type]],
                [pair[1] for pair in confidence_scores_merged[gateway_type]],
                label=gateway_type, color=color, marker='.')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Gateway size (number of activities in all branches)")
    plt.ylabel("Avg. confidence score")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(ROOT_DIR, "data/paper_stats/activity_relation/gateway_extraction", plot_name))


if __name__ == '__main__':
    plot_confidence_size_correlation("data/results_relation_approaches/gateway_extraction/ge=standard_rc=goldstandard_vote=full/predictions.json",
                                     "GE_rc=goldstandard_confidence_plot")


