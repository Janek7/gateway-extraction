import json
import os

def list_runs(root_dir, order_by=None):
    run_subdirs = [x[0] for x in os.walk(root_dir)]
    print(run_subdirs[1])

    # create dict with param string as key and related cv_metrics dict as value
    runs = []
    for sub_dir in run_subdirs:
        if sub_dir.endswith("False"):
            # cut generic parts & parse
            bs_index = sub_dir.index("bs")
            param_list_short = sub_dir[bs_index:]
            # read cv_metrics.json
            with open(os.path.join(sub_dir, "cv_metrics.json"), "r") as file:
                cv_metrics = json.load(file)
            relevant_metrics = {k: v for k, v in cv_metrics.items() if k.startswith("avg")}
            runs.append((param_list_short, relevant_metrics))

    if order_by:
        runs.sort(key=lambda params_metrics_tuple: params_metrics_tuple[1][order_by], reverse=True)

    return runs

if __name__ == '__main__':
    runs = list_runs("/data/logs_server/newest_consistency_check",
                     order_by="avg_val_xor_recall")
    for params, metrics in runs:
        print(params)
        print(metrics)
        print()