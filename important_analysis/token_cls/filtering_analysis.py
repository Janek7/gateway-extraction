# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import json
import pandas as pd

from utils import ROOT_DIR, load_pickle
from PetReader import pet_reader

KEPT = 'kept'
CORRECTLY_KEPT = 'correctly_kept'
WRONGLY_KEPT = 'wrongly_kept'
DROPPED = 'dropped'
CORRECTLY_DROPPED = 'correctly_dropped'
WRONGLY_DROPPED = 'wrongly_dropped'


def analyze_filterings(filterings_filename: str, approach_name: str) -> None:
    # 1) load data
    with open(os.path.join(ROOT_DIR, filterings_filename), "r") as file:
        filtering_file = json.load(file)

    statistics = {}

    # 2) compute stats
    for doc_name, filterings in filtering_file.items():
        doc_gold = pet_reader.relations_dataset.GetSentencesWithIdsAndNerTagLabels(
            pet_reader.get_document_number(doc_name))
        for sentence_idx, tokens in enumerate(doc_gold):
            # print(sentence_idx, tokens)
            sentence_filterings = [f for f in filterings if f[0] == sentence_idx]
            for f in sentence_filterings:
                t = f[1][0].lower()
                if t not in statistics:
                    statistics[t] = {KEPT: 0, CORRECTLY_KEPT: 0, WRONGLY_KEPT: 0, DROPPED: 0, CORRECTLY_DROPPED: 0,
                                     WRONGLY_DROPPED: 0}
                # token with predicted class IN tokens
                if tuple(f[1]) in tokens:
                    # kept
                    if f[3] == KEPT:
                        statistics[t][KEPT] += 1
                        statistics[t][CORRECTLY_KEPT] += 1
                    # dropped
                    else:
                        statistics[t][DROPPED] += 1
                        statistics[t][WRONGLY_DROPPED] += 1
                # token with predicted class NOT IN tokens
                else:
                    # dropped
                    if f[3] == DROPPED:
                        statistics[t][DROPPED] += 1
                        statistics[t][CORRECTLY_DROPPED] += 1
                    # kept
                    else:
                        statistics[t][KEPT] += 1
                        statistics[t][WRONGLY_KEPT] += 1

    # 3) compute more stats
    rows = []
    for keyword, stats in statistics.items():
        print(keyword, stats)
        kept_ratio = stats[KEPT] / (stats[KEPT] + stats[DROPPED])
        dropped_ratio = stats[DROPPED] / (stats[KEPT] + stats[DROPPED])
        stats["kept/dropped"] = f"{round(kept_ratio, 2)} / {round(dropped_ratio, 2)}"
        stats["correct"] = stats[CORRECTLY_KEPT] + stats[CORRECTLY_DROPPED]
        stats["wrong"] = stats[WRONGLY_KEPT] + stats[WRONGLY_DROPPED]
        rows.append({**{"word": keyword}, **stats})

    # 4) export to xlsx
    df = pd.DataFrame.from_dict(rows)
    print(df.head(10))
    df.to_excel(os.path.join(ROOT_DIR, f"data/paper_stats/token_cls/filtering_stats_{approach_name}.xlsx"), index=False)


if __name__ == '__main__':
    analyze_filterings(
        "data/results/keywords_filtered/key_words_literature_tc_filtered_og_syn/filtering_XOR Gateways.json",
        approach_name="literature")
