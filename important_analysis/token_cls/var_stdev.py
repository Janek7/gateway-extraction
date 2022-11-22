import json
import os
from statistics import stdev, variance, mean

import pandas as pd

from utils import ROOT_DIR


def compute_seed_stats():
    """
    computes statistics about how metrics differ between different seeds
    example is taken from one fold
    :return:
    """
    example_fold = 0
    report_metrics = ['val_xor_precision', 'val_xor_recall', 'val_xor_f1',
                      'val_and_precision', 'val_and_recall', 'val_and_f1',
                      'val_overall_accuracy']
    with open(os.path.join(ROOT_DIR, "data\logs_server\FINAL_example_differences\GatewayTokenClassifier_train.py-2022-11-17_112658-au=not,bs=8,e=True,e=1,f=5,l=all,olw=0.1,r=cv,ss=normal,sg=42,se=0-29,sw=False,us=False\cv_metrics.json"),
              "r") as file:
        cv_metrics = json.load(file)
    rows = []
    for metric in report_metrics:
        values = cv_metrics[f"seed-results-{metric}-{example_fold}"]
        rows.append({"metric": metric,
                     "mean": mean(values),
                     "variance": variance(values),
                     "stdev": stdev(values),
                     "max": max(values),
                     "min": min(values)})

    # output to dataframe
    df = pd.DataFrame.from_dict(rows)
    print(df.head(10))
    df.to_excel("../data/paper_stats/seed_differences.xlsx",sheet_name="seeds=0-29", index=False)


def compute_gpu_run_diffs():
    # read cv metric files from runs in dir
    runs = []
    for sub_dir in [x[0] for x in os.walk("/data/logs_server/_final/FINAL_example_differences")]:
        if sub_dir.endswith("False"):
            with open(os.path.join(sub_dir, "cv_metrics.json"), "r") as file:
                cv_metrics = json.load(file)
                runs.append(cv_metrics)

    report_metrics = ['avg_val_xor_precision', 'avg_val_xor_recall', 'avg_val_xor_f1',
                      'avg_val_and_precision', 'avg_val_and_recall', 'avg_val_and_f1',
                      'avg_val_overall_accuracy']

    # collect values for target metrics from each run
    rows = []
    metrics_dict = {m: [] for m in report_metrics}
    for metric in report_metrics:
        for cv_metrics in runs:
            print(cv_metrics.keys())
            metrics_dict[metric].append(cv_metrics[metric])

    # compute statistics
    for metric, values in metrics_dict.items():
        rows.append({"metric": metric,
                     "mean": mean(values),
                     "variance": variance(values),
                     "stdev": stdev(values),
                     "max": max(values),
                     "min": min(values)})

    # output to dataframe
    df = pd.DataFrame.from_dict(rows)
    print(df.head(10))
    df.to_excel(os.path.join(ROOT_DIR, "/data/paper_stats/gpu_run_differences.xlsx"), sheet_name="6 runs", index=False)


if __name__ == '__main__':
    # compute_seed_stats()
    compute_gpu_run_diffs()
