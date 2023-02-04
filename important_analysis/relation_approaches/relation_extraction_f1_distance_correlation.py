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
import pandas as pd
import matplotlib.pyplot as plt

from utils import ROOT_DIR
from labels import *

Ns = [1, 2, 5, 10, 30]


def plot_f1_nearest_n_correlation(results_folder: str, approach_name: str) -> None:
    """
    creates a graph that plots for each label on y f1 scores and on x the nearest n relations
    :param results_folder: root dir relative path to output folder with all the result files
    :param approach_name: approach_name
    :return: image output to directory
    """
    # load data
    nearest_n_stats_dfs = {n: pd.read_excel(os.path.join(ROOT_DIR, results_folder, f"results-n={n}.xlsx"),
                                            sheet_name="Label-wise metrics")
    if n != 30 else pd.read_excel(os.path.join(ROOT_DIR, results_folder, f"results-all.xlsx"),
                                  sheet_name="Label-wise metrics")
                           for n in Ns}
    graph_data = {DIRECTLY_FOLLOWING: [], EVENTUALLY_FOLLOWING: [], EXCLUSIVE: [], CONCURRENT: []}
    for n, stats_df in nearest_n_stats_dfs.items():
        for label in graph_data:
            graph_data[label].append((n, stats_df[stats_df["label"] == label]["f1-score"]))
        print(n)
        print(stats_df.head(10))

    # config output
    relation_type_data = [(DIRECTLY_FOLLOWING, "lightcoral", "Directly Following"),
                          (EVENTUALLY_FOLLOWING, "khaki", "Eventually Following"),
                          (EXCLUSIVE, "cornflowerblue", "Exclusive"),
                          (CONCURRENT, "mediumseagreen", "Concurrent")]

    # plot
    fig, ax = plt.subplots()

    for gateway_type, color, label in relation_type_data:
        ax.plot([n for n, f1 in graph_data[gateway_type]],
                [f1 for n, f1 in graph_data[gateway_type]],
                label=label, color=color, marker='.')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Activity order distance")
    plt.ylabel("Avg. F1 score")
    plt.xticks([1, 2, 5, 10, 30], [1, 2, 5, 10, "all"])

    plt.legend(loc="center right")
    plt.savefig(os.path.join(ROOT_DIR,
                             f"data/paper_stats/activity_relation/gateway_extraction/RC_nearest_n_plot_{approach_name}"))


if __name__ == '__main__':
    plot_f1_nearest_n_correlation("data/results_relation_approaches/relation_classification/brcnn_128",
                                  "brcnn128")
