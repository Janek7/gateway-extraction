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
from typing import Dict

from petreader.labels import *
import numpy as np

from utils import ROOT_DIR


def average_confidence_scores(gateway_extractions_file: str) -> Dict[str, float]:
    """
    average confidence score of gateways with passed type in given file
    :param gateway_extractions_file: root dir relative path to json file with extractions
    :return: avg confidence score
    """
    with open(os.path.join(ROOT_DIR, gateway_extractions_file)) as file:
        extractions = json.load(file)
    confidence_scores = {XOR_GATEWAY: [], AND_GATEWAY: []}
    for doc_name, gateways in extractions.items():
        for gateway in gateways:
            if gateway["type"] == XOR_GATEWAY:
                confidence_scores[XOR_GATEWAY].append(gateway["confidence"])
            elif gateway["type"] == AND_GATEWAY:
                confidence_scores[AND_GATEWAY].append(gateway["confidence"])

    return {gateway_type: np.mean(confidence_scores) for gateway_type, confidence_scores in confidence_scores.items()}


if __name__ == '__main__':
    full_vote = average_confidence_scores("data/results_relation_approaches/gateway_extraction/ge=standard_rc=goldstandard_vote=full_TESTDOCS/predictions.json")
    print(f"full_vote: {full_vote}")

    limited_vote = average_confidence_scores("data/results_relation_approaches/gateway_extraction/ge=standard_rc=goldstandard_vote=limited_TESTDOCS/predictions.json")
    print(f"limited_vote: {limited_vote}")
