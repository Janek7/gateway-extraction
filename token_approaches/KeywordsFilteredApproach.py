#!/usr/bin/env python3

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
import os.path
from typing import List, Tuple

from petreader.labels import *
import tensorflow as tf
# fix for exception "Attempting to perform BLAS operation using StreamExecutor without BLAS support"
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from GatewayTokenClassifier import GatewayTokenClassifier, convert_predictions_into_labels
from Ensemble import Ensemble
from KeywordsApproach import KeywordsApproach
from PetReader import pet_reader
from token_data_preparation import preprocess_tokenization_data
from utils import config, set_seeds, NumpyEncoder
from labels import *

logger = logging.getLogger('Keywords Filtered Approach')


class KeywordsFilteredApproach(KeywordsApproach):
    """
    extend KeywordsApproach by filtering with GatewayTokenClassifier
    """

    def __init__(self, approach_name: str = None, keywords: str = LITERATURE, contradictory_keywords: str = GOLD,
                 same_xor_gateway_threshold: int = 1, multiple_branches_allowed: bool = False,
                 output_format: str = BENCHMARK, output_folder: str = None,
                 xor_rule_c: bool = True, xor_rule_or: bool = True, xor_rule_op: bool = True,
                 # class / ensemble specific params:
                 ensemble_path: str = None, seed_limit: int = None, mode: str = DROP, filtering_log_level: str = FILE):
        """
        creates new instance of the advanced keywords filtered approach
        ---- super class params ----
        :param approach_name: description of approach to use in result folder name; if not set use key word variant
        :param keywords: flag/variant which keywords to use; available: literature, gold, own
        :param same_xor_gateway_threshold: threshold to recognize subsequent (contradictory xor) gateways as same
        :param output_format: output format of extracted element and flows; available: benchmark, pet
        :param output_folder: name of output folder; if none -> create based on approach name
        :param xor_rule_c: flag if rule for detection of contradictory gateways should be applied
        :param xor_rule_or: flag if rule for detection of 'or' gateways should be applied
        :param xor_rule_op: flag if rule for detection of one branch (optional branches) should be applied
        -- class / ensemble params ---
        :param ensemble_path: path of ensemble model to restore weights from;
                              if None, a random initialized model will be used
        :param seed_limit: limit of seeds to reload from the ensemble (in case of OOM errors)
        :param mode: filter mode: 'log' to only log difference; 'drop' to drop gateways with diff. token cls prediction
        :param filtering_log_level: 'file', 'console' or None
        """
        super().__init__(approach_name=approach_name, keywords=keywords, contradictory_keywords=contradictory_keywords,
                         same_xor_gateway_threshold=same_xor_gateway_threshold,
                         multiple_branches_allowed=multiple_branches_allowed, output_format=output_format,
                         output_folder=output_folder,
                         xor_rule_c=xor_rule_c, xor_rule_or=xor_rule_or, xor_rule_op=xor_rule_op)
        self.token_classifier = Ensemble(args=None, model_class=GatewayTokenClassifier, ensemble_path=ensemble_path,
                                         seed_limit=seed_limit)
        set_seeds(config[SEED], "Reset after initialization of GatewayTokenClassifierEnsemble")
        self.mode = mode
        self.filtering_log_level = filtering_log_level

    def filter_gateways(self, doc_name: str, xor_gateways: List[List[Tuple[str, int, str]]],
                        and_gateways: List[List[Tuple[str, int, str]]]) \
            -> Tuple[List[List[Tuple[str, int, str]]], List[List[Tuple[str, int, str]]]]:
        """
        filter given gateways of given documents using the GatewayTokenClassifier model
        :param doc_name: name of document to which gateways belongs
        :param xor_gateways: list of xor gateways of target document in PET format
        :param and_gateways: list of and gateways of target document in PET format
        :return: xor gateways, and gateways (same format, just filtered)
        """
        # preprocess data
        tokens, _, _, word_ids = preprocess_tokenization_data(sample_numbers=pet_reader.get_doc_sample_ids(doc_name))

        # predict token labels with GatewayTokenClassifier
        predictions = self.token_classifier.predict(tokens)
        predictions = convert_predictions_into_labels(predictions, word_ids)

        # filter gateway lists using predictions
        def filter_gateways(gateways, gateway_type):
            gateway_type_token_label = TC_LABEL_XOR if gateway_type == XOR_GATEWAY else TC_LABEL_AND
            log_messages = [f" {doc_name} / {gateway_type} ".center(50, '-')]
            log_objects = []
            filtered_gateways = []
            for i, (sentence_gateways, sentence_token_predictions) in enumerate(zip(gateways, predictions)):
                filtered_sentence_gateways = []
                for g in sentence_gateways:
                    if g[2].endswith(gateway_type):
                        log_object = [i, g, sentence_token_predictions[g[1]]]
                        # check if keyword extracted gateway is not predicted as gateway in token classification
                        if sentence_token_predictions[g[1]] != gateway_type_token_label:
                            log_msg = f"dropped [sent={i}] {gateway_type} {g} " \
                                      f"-> token mismatch: {sentence_token_predictions[g[1]]}"
                            log_object.append("dropped")
                            if self.mode == LOG:
                                filtered_sentence_gateways.append(g)
                            elif self.mode == DROP:
                                pass  # do not add g to filtered list
                        else:
                            log_msg = f"kept [sent={i}] {gateway_type} {g} " \
                                      f"-> token match: {sentence_token_predictions[g[1]]}"
                            filtered_sentence_gateways.append(g)
                            log_object.append("kept")
                        log_messages.append(log_msg)
                        log_objects.append(log_object)
                filtered_gateways.append(filtered_sentence_gateways)

            # write filter logs into log file (when it exists -> it only exists in when evaluating all documents)
            if self.filtering_log_level == FILE:
                if os.path.isdir(self.results_folder):
                    # 1) txt log file
                    with open(os.path.join(self.results_folder, f"filtering_{gateway_type}s.txt"), "a") as file:
                        file.write('\n'.join(log_messages) + '\n')

                    # 2) json data file
                    filename = os.path.join(self.results_folder, f"filtering_{gateway_type}s.json")
                    if os.path.isfile(filename):
                        with open(filename, "r+") as file:
                            content = json.load(file)
                    else:
                        content = {}
                    with open(filename, "w") as file:
                        content[doc_name] = log_objects
                        json.dump(content, file, indent=4, cls=NumpyEncoder)

                else:
                    logger.warning("Can not write to result file when evaluating single documents")

            # write filter logs into console
            elif self.filtering_log_level == CONSOLE:
                for msg in log_messages:
                    logger.info(msg)

            return filtered_gateways

        filtered_xor_gateways = filter_gateways(xor_gateways, XOR_GATEWAY)
        filtered_and_gateways = and_gateways  # filter_gateways(and_gateways, AND_GATEWAY)
        return filtered_xor_gateways, filtered_and_gateways


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    set_seeds(config[SEED], "Set first seed")
    keyword_filtered_approach = KeywordsFilteredApproach(approach_name='key_words_literature_tc_filtered_og_syn',
                                                         # three cases to evaluate with filter model
                                                         keywords=LITERATURE,
                                                         # keywords=CUSTOM,
                                                         # keywords=CUSTOM, contradictory_keywords=GOLD, same_xor_gateway_threshold=3, multiple_branches_allowed=True, seed_limit=15,

                                                         ensemble_path="/home/japutz/master-thesis/data/final_models/token_cls/GatewayTokenClassifier_train.py-2022-11-19_074241-au=not,bs=8,e=True,e=1,f=2,l=all,olw=0.1,r=ft,ss=og,sg=42,se=0-29,sw=True,us=True",
                                                         mode=DROP, filtering_log_level=FILE)
    if True:
        keyword_filtered_approach.evaluate_documents(evaluate_token_cls=True, evaluate_relation_extraction=True)
    if False:
        doc_name = 'doc-3.2'
        xor_gateways, and_gateways, doc_flows, same_gateway_relations = keyword_filtered_approach.process_document(doc_name)

        print(" Concurrent gateways ".center(50, '-'))
        for gateway in and_gateways:
            print(gateway)

        print(" Exclusive gateways ".center(50, '-'))
        for gateway in xor_gateways:
            print(gateway)