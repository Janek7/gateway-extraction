#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import logging
from typing import List, Tuple

import tensorflow as tf

# fix for exception "Attempting to perform BLAS operation using StreamExecutor without BLAS support"
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from token_approaches.SameGatewayClassifier import SGCEnsemble
from token_approaches.KeywordsApproach import KeywordsApproach
from utils import config, set_seeds
from labels import *

logger = logging.getLogger('Keywords Same Gateway Filtered Approach')


class KeywordsSGCApproach(KeywordsApproach):
    """
    extend KeywordsApproach by evaluating same gateway relations with model
    """

    def __init__(self, approach_name: str = None, keywords: str = LITERATURE,
                 output_format: str = BENCHMARK, output_folder: str = None,
                 xor_rule_c: bool = True, xor_rule_or: bool = True, xor_rule_op: bool = True,
                 # class / ensemble specific params:
                 ensemble_path: str = None, seed_limit: int = None):
        """
        creates new instance of the same gateway relation classification approach
        ---- super class params ----
        :param approach_name: description of approach to use in result folder name; if not set use key word variant
        :param keywords: flag/variant which keywords to use; available: literature, gold, own
        :param output_format: output format of extracted element and flows; available: benchmark, pet
        :param output_folder: name of output folder; if none -> create based on approach name
        :param xor_rule_c: flag if rule for detection of contradictory gateways should be applied
        :param xor_rule_or: flag if rule for detection of 'or' gateways should be applied
        :param xor_rule_op: flag if rule for detection of one branch (optional branches) should be applied
        -- class / ensemble params ---
        :param ensemble_path: path of ensemble model to restore weights from;
                              if None, a random initialized model will be used
        :param seed_limit: limit of seeds to reload from the ensemble (in case of OOM errors)
        """
        super().__init__(approach_name=approach_name, keywords=keywords,
                         output_format=output_format, output_folder=output_folder,
                         xor_rule_c=xor_rule_c, xor_rule_or=xor_rule_or, xor_rule_op=xor_rule_op)
        self.same_gateway_classifier = SGCEnsemble(args=None, ensemble_path=ensemble_path, seed_limit=seed_limit)
        set_seeds(config[SEED], "Reset after initialization of SameGatewayClassifierEnsemble")

    def extract_same_gateway_pairs(self, doc_name: str, gateways: List[Tuple], gateways_involved_contradictory: List):
        """
        extracts a list of same gateway relations from a list of subsequent gateways using a NEURAL CLASSIFIER
        :param doc_name: document name
        :param gateways: list of gateways
        :param gateways_involved_contradictory: temp list of gateways already involved into a contradictory gateway
        :return: same gateway relations as a list of gateway relations
        """
        same_gateway_pairs = []
        for i in range(len(gateways) - 1):
            g1, g2 = gateways[i], gateways[i + 1]
            if self.same_gateway_classifier.classify_pair_bool(doc_name, g1[ELEMENT], g2[ELEMENT]):
                same_gateway_pairs.append((g1, g2))
        for g1, g2 in same_gateway_pairs:
            print(g1[ELEMENT], ",", g2[ELEMENT])
        return same_gateway_pairs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    set_seeds(config[SEED], "Set first seed")

    # two cases to evaluate with sg classification model
    test_cases = [('literature', LITERATURE), ('custom', CUSTOM)]

    for approach_name, keywords in test_cases:
        keyword_filtered_approach = KeywordsSGCApproach(
            # params of keyword approach
            approach_name=f'key_words_{approach_name}_sg_classified_[context_text_labels_ngram_c1_n0_syn]',
            keywords=keywords,
            # params of token cls model
            ensemble_path="/home/japutz/master-thesis/data/final_models/SameGatewayClassifier-2023-01-05_075226-am=not,bs=8,cs=1,d=0.2,e=True,e=10,f=2,g=XOR Gateway,hl=32,lr=2e-05,m=context_text_and_labels_n_gram,ng=0,r=ft,sg=42,se=10-20,sw=True,us=True,w=0"
        )
        keyword_filtered_approach.evaluate_documents(evaluate_token_cls=True, evaluate_relation_extraction=True)


    if False:
        doc_name = 'doc-3.2'
        xor_gateways, and_gateways, doc_flows, same_gateway_relations = keyword_filtered_approach.process_document(
            doc_name)

        print(" Concurrent gateways ".center(50, '-'))
        for gateway in and_gateways:
            print(gateway)

        print(" Exclusive gateways ".center(50, '-'))
        for gateway in xor_gateways:
            print(gateway)
