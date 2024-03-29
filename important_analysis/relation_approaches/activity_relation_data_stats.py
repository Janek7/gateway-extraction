# add parent dir to sys path for import of modules
import json
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import logging

import pandas as pd

from labels import *
from relation_approaches.activity_relation_data_preparation import get_activity_relations
from utils import ROOT_DIR


def _create_statistics():
    relations = get_activity_relations(return_type=dict)

    for r in relations:
        r["same_sentence"] = r[ACTIVITY_1][0] == r[ACTIVITY_2][0]

    df = pd.DataFrame.from_dict(relations)
    df.fillna("", inplace=True)

    relation_type_count = df.groupby(RELATION_TYPE).count()
    print(relation_type_count)
    print(100 * '-')
    relation_type_same_sentence_count = df.groupby(["same_sentence", RELATION_TYPE]).count()
    print(relation_type_same_sentence_count)
    print(100 * '-')
    relation_type_comment_count = df.groupby([RELATION_TYPE, COMMENT]).count()
    print(relation_type_comment_count)

    # counts doc stats
    print(100 * '-')
    doc_count = df.groupby(DOC_NAME).count()
    print(doc_count)
    print(100 * '-')
    doc_stats = df.groupby(DOC_NAME).count().describe()
    print(doc_stats)

    with pd.ExcelWriter(os.path.join(ROOT_DIR, 'data/paper_stats/activity_relation/data_stats/activity_relation_data_stats.xlsx')) \
            as writer:
        relation_type_count.to_excel(writer, sheet_name='Relation Type Count')
        relation_type_same_sentence_count.to_excel(writer, sheet_name='Same Sentence Count')
        relation_type_comment_count.to_excel(writer, sheet_name='Comment Count')
        doc_count.to_excel(writer, sheet_name='Doc Count')
        doc_stats.to_excel(writer, sheet_name='Doc Stats')


def _analyze_nested_gateways():
    with open(os.path.join(ROOT_DIR, "data/paper_stats/activity_relation/data_stats/nested_gateways.json"), 'r') as file:
        nested_gateways = json.load(file)["nested_gateways"]
    df = pd.DataFrame.from_dict(nested_gateways)

    print("-"*100)
    df["parent_str"] = [str(x) for x in df["parent"]]
    parent_stats = df.groupby(["doc_name", "parent_str"]).count().sort_values("parent", ascending=False)
    print(parent_stats.head(30))
    print("-" * 100)
    print(f"{parent_stats.count()} have nested gateways inside")

    print("-" * 100)

    with pd.ExcelWriter(os.path.join(ROOT_DIR, 'data/paper_stats/activity_relation/data_stats/nested_gateways.xlsx')) as writer:
        df.to_excel(writer, sheet_name='Nested Gateways', index=False)
        parent_stats.to_excel(writer, sheet_name='Nested Gateways Grouped')


def _analyze_branch_lengths():
    with open(os.path.join(ROOT_DIR, "data/paper_stats/activity_relation/data_stats/branch_lengths.json"),'r') as file:
        branch_lengths = json.load(file)["branch_lengths"]
    df = pd.DataFrame.from_dict(branch_lengths)

    stats = df.describe()
    print(stats)

    with pd.ExcelWriter(os.path.join(ROOT_DIR, 'data/paper_stats/activity_relation/data_stats/branch_lengths.xlsx')) as writer:
        df.to_excel(writer, sheet_name='Branch lengths', index=False)
        stats.to_excel(writer, sheet_name='Branch lengths stats')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _create_statistics()
    _analyze_nested_gateways()
    _analyze_branch_lengths()
