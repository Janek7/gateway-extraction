# add parent dir to sys path for import of modules
import json
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

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
    evaluation_df = pd.read_excel(os.path.join(ROOT_DIR, results_folder, f"results.xlsx"), sheet_name="Evaluation")
    evaluation_df.info()
    print(evaluation_df.head(10))
    graph_data = {DIRECTLY_FOLLOWING: [], EVENTUALLY_FOLLOWING: [], EXCLUSIVE: [], CONCURRENT: [],} #  "all": []
    for label in graph_data:
        for n in Ns:
            print(evaluation_df[(evaluation_df["label"] == label) & (evaluation_df["n"] == n)]["f1"].values[0])
            graph_data[label].append(
                (n, evaluation_df[(evaluation_df["label"] == label) & (evaluation_df["n"] == n)]["f1"].values[0]))

    # config output
    relation_type_data = [
        # ("all", "silver", "All"),
                          (DIRECTLY_FOLLOWING, "lightcoral", "Directly Following"),
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
    plt.xlabel("Activity pair distance <= n")
    plt.ylabel("Avg. F1 score")
    plt.xticks([1, 2, 5, 10, 30], [1, 2, 5, 10, "all"])
    plt.yticks([.1, .2, .3, .4, .5, .6, .7, .8, .9], [.1, .2, .3, .4, .5, .6, .7, .8, .9])

    plt.legend(loc="center right")
    plt.savefig(os.path.join(ROOT_DIR,
                             f"data/paper_stats/activity_relation/relation_classification/nearest_n_plot_{approach_name}"))


if __name__ == '__main__':
    plot_f1_nearest_n_correlation(
        "data/results_relation_approaches/relation_classification/brcnn_128",
        "brcnn_128")
