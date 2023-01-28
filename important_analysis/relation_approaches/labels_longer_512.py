# add parent dir to sys path for import of modules
import json
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

from collections import Counter

from utils import ROOT_DIR

with open(os.path.join(ROOT_DIR, "data/paper_stats/activity_relation/labels_from_longer_512.txt"), "r") as file:
    labels = [int(line) for line in file.readlines()]
    print(len(labels))
    print(Counter(labels))
