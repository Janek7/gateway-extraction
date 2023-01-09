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
from petreader.labels import *

# fix for exception "Attempting to perform BLAS operation using StreamExecutor without BLAS support"
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from PetReader import pet_reader
from token_approaches.SameGatewayClassifier import SGCEnsemble
from token_approaches.KeywordsApproach import KeywordsApproach
from utils import config, set_seeds
from labels import *

logger = logging.getLogger('Keywords Same Gateway Filtered Approach')


class KeywordsSGCApproach(KeywordsApproach):
    """
    extend KeywordsApproach by evaluating same gateway relations with model
    """

    def __init__(self, approach_name: str = None, blacklist_or: bool = True, distance_threshold: int = 3,
                 keywords: str = LITERATURE, output_format: str = BENCHMARK, output_folder: str = None,
                 xor_rule_c: bool = True, xor_rule_or: bool = True, xor_rule_op: bool = True,
                 # class / ensemble specific params:
                 ensemble_path: str = None, seed_limit: int = None):
        """
        creates new instance of the same gateway relation classification approach
        ---- super class params ----
        :param blacklist_or: flag if gateway pairs including 'or' should be always classified as non-related
        :param distance_threshold: sentence distance threshold above gateway pairs from should be always classified as
                                   not same gateway
        ---- super class params ----
        :param approach_name: description of approach to use in result folder name; if not set use key word variant
        :param keywords: flag/variant which keywords to use; available: literature, gold, own
        :param output_format: output format of extracted element and flows; available: benchmark, pet
        :param output_folder: name of output folder; if none -> create based on approach name
        :param xor_rule_c: flag if rule for detection of contradictory gateways should be applied
        :param xor_rule_or: flag if rule for detection of 'or' gateways should be applied
        :param xor_rule_op: flag if rule for detection of one branch (optional branches) should be applied
        -- ensemble params ---
        :param ensemble_path: path of ensemble model to restore weights from;
                              if None, a random initialized model will be used
        :param seed_limit: limit of seeds to reload from the ensemble (in case of OOM errors)
        """
        super().__init__(approach_name=approach_name, keywords=keywords,
                         output_format=output_format, output_folder=output_folder,
                         xor_rule_c=xor_rule_c, xor_rule_or=xor_rule_or, xor_rule_op=xor_rule_op)
        self.same_gateway_classifier = SGCEnsemble(args=None, log_folder=self.results_folder,
                                                   ensemble_path=ensemble_path, seed_limit=seed_limit)
        set_seeds(config[SEED], "Reset after initialization of SameGatewayClassifierEnsemble")
        self.blacklist_or = blacklist_or
        self.distance_threshold = distance_threshold

    def extract_same_gateway_pairs(self, doc_name: str, gateways: List[Tuple], gateways_involved_contradictory: List):
        """
        extracts a list of same gateway relations from a list of subsequent gateways using a NEURAL CLASSIFIER
        :param doc_name: document name
        :param gateways: list of gateways in internal representation of KeywordsApproach
        :param gateways_involved_contradictory: temp list of gateways already involved into a contradictory gateway
        :return: same gateway relations as a list of gateway relations
        """
        same_gateway_pairs = []
        for i in range(len(gateways) - 1):
            g1, g2 = gateways[i], gateways[i + 1]
            # if sentence distance is greater than threshold -> do not classify as same gateway pair
            if self.distance_threshold and abs(g1[ELEMENT][0] - g2[ELEMENT][0]) > self.distance_threshold:
                self.same_gateway_classifier.log_prediction(doc_name, g1[ELEMENT], g2[ELEMENT], 0, [0],
                                                            comment="rule: sentence distance > 3")
            # if phrase pair is listed in blacklist -> do not classify as same gateway pair
            elif self.blacklist_or and ['or'] in [g1[ELEMENT][3], g2[ELEMENT][3]]:
                self.same_gateway_classifier.log_prediction(doc_name, g1[ELEMENT], g2[ELEMENT], 0, [0],
                                                            comment="rule: involves 'or'")
            elif self.same_gateway_classifier.classify_pair_bool(doc_name, g1[ELEMENT], g2[ELEMENT]):
                same_gateway_pairs.append((g1, g2))
        # for g1, g2 in same_gateway_pairs:
        #     logger.info(g1[ELEMENT], ",", g2[ELEMENT])
        return same_gateway_pairs


class GoldGatewaysSGCApproach(KeywordsSGCApproach):
    """
    extend KeywordsSGCApproach by using gold data for gateway tokens to see what would be possible with trained SGC
    when it is based on gold token/gateway data
    """

    def filter_gateways(self, doc_name: str, xor_gateways: List[List[Tuple[str, int, str]]],
                        and_gateways: List[List[Tuple[str, int, str]]]) \
            -> Tuple[List[List[Tuple[str, int, str]]], List[List[Tuple[str, int, str]]]]:
        """
        abuse overwrite of filter method to return gold gateways
        """
        get_gateway_tokens = lambda doc_name: [[t for t in s if t[2].endswith(XOR_GATEWAY)] for s in
                                               pet_reader.get_ner_tags(pet_reader.get_document_number(doc_name))]
        return get_gateway_tokens(XOR_GATEWAY), get_gateway_tokens(AND_GATEWAY)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    set_seeds(config[SEED], "Set first seed")

    # two cases to evaluate with sg classification model
    test_cases = [#(KeywordsSGCApproach, 'literature', LITERATURE),
                  #(KeywordsSGCApproach, 'custom', CUSTOM),
                  # keywords here only dummy -> overwritten by filter_gateways
                  (GoldGatewaysSGCApproach, 'custom', CUSTOM)]

    for approach_class, approach_name, keywords in test_cases:
        keyword_filtered_approach = approach_class(
            # params of keyword approach
            approach_name=f'key_words_{approach_name}_sg_classified_rules_[e5_context_text_labels_ngram_c1_n0_syn]',
            keywords=keywords,
            # if commented -> with rules, if not commented and params active -> without rules
            #blacklist_or=False,
            #distance_threshold=None,
            # params of same gateway ensemble model
            ensemble_path="/home/japutz/master-thesis/data/final_models/SameGatewayClassifier-2023-01-05_091133-am=not,bs=8,cs=1,d=0.2,e=True,e=5,f=2,g=XOR Gateway,hl=32,lr=2e-05,m=context_text_and_labels_n_gram,ng=0,r=ft,sg=42,se=10-20,sw=True,us=True,w=0"
        )
        keyword_filtered_approach.evaluate_documents(evaluate_token_cls=True, evaluate_relation_extraction=True)
        keyword_filtered_approach.same_gateway_classifier.save_prediction_logs()

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
