import json
import os

import pandas as pd

from utils import ROOT_DIR


def compare_runs(root_dir, name, model_type, order_by=None):
    run_subdirs = [x[0] for x in os.walk(root_dir)]

    # create dict with param string as key and related cv_metrics dict as value
    runs = []
    for sub_dir in run_subdirs:
        if "GatewayTokenClassifier" in sub_dir or "SameGatewayClassifier" in sub_dir:
            print(sub_dir)
            # cut generic parts & parse
            bs_index = sub_dir.index("am=")
            param_list_short = sub_dir[bs_index:]
            # read cv_metrics.json
            with open(os.path.join(sub_dir, "cv_metrics.json"), "r") as file:
                cv_metrics = json.load(file)
            relevant_metrics = {k: v for k, v in cv_metrics.items() if k.startswith("avg")}
            runs.append({**{"params": param_list_short}, **relevant_metrics})

    if order_by:
        runs.sort(key=lambda dict: dict[order_by], reverse=True)

    # save results
    df = pd.DataFrame.from_dict(runs)
    print(df.head(100))
    df.to_excel(os.path.join(ROOT_DIR, f"data/paper_stats/{model_type}/run_results_{name}.xlsx"),
                sheet_name="seeds=0-29", index=False)


if __name__ == '__main__':
    # for name in ['n_gram', 'index', 'concat']:
    #     compare_runs(os.path.join(ROOT_DIR, f"data/logs_server/same_gateway/{name}"),
    #                  order_by="avg_val_recall",
    #                  name=name)
    # compare_runs(os.path.join(ROOT_DIR, f"data/logs_server/same_gateway"),
    #              order_by="avg_val_recall",
    #              name="all")
    compare_runs(os.path.join(ROOT_DIR, f"data/logs_server/same_gateway/context_text_labels_n_gram"),
                 name="context_text_labels_n_gram",
                 model_type="same_gateway_cls",  # token_cls or same_gateway_cls
                 order_by="avg_val_recall")
