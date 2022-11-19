import logging
import os.path
from typing import List, Tuple

import tensorflow as tf
from petreader.labels import *

from GatewayTokenClassifier import GatewayTokenClassifier, GatewayTokenClassifierEnsemble
from KeywordsApproach import KeywordsApproach
from PetReader import pet_reader
from token_data_preparation import preprocess_tokenization_data
from utils import config, set_seeds
from labels import *

logger = logging.getLogger('Keywords Filtered Approach')


class KeywordsFilteredApproach(KeywordsApproach):
    """
    extend KeywordsApproach by filtering with GatewayTokenClassifier
    """

    def __init__(self, approach_name: str = None, keywords: str = LITERATURE, output_format: str = BENCHMARK,
                 same_xor_gateway_threshold: int = 1, output_folder: str = None,
                 # class specific params:
                 ensemble_path: str = None, mode: str = DROP, filtering_log_level: str = FILE):
        """
        creates new instance of the advanced keywords filtered approach
        :param approach_name: description of approach to use in result folder name; if not set use key word variant
        :param keywords: flag/variant which keywords to use; available: literature, gold, own
        :param same_xor_gateway_threshold: threshold to recognize subsequent (contradictory xor) gateways as same
        :param output_format: output format of extracted element and flows; available: benchmark, pet
        :param output_folder: name of output folder; if none -> create based on approach name
        :param ensemble_path: path of ensemble model to restore weights from;
                              if None, a random initialized model will be used
        :param mode: filter mode: 'log' to only log difference; 'drop' to drop gateways with diff. token cls prediction
        :param filtering_log_level: 'file', 'console' or None
        """
        super().__init__(approach_name=approach_name, keywords=keywords, output_format=output_format,
                         same_xor_gateway_threshold=same_xor_gateway_threshold, output_folder=output_folder)
        self.token_classifier = GatewayTokenClassifierEnsemble(ensemble_path=ensemble_path)
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
        predictions = self.token_classifier.predict_converted(tokens, word_ids)

        # filter gateway lists using predictions
        def filter_gateways(gateways, gateway_type):
            gateway_type_token_label = TC_LABEL_XOR if gateway_type == XOR_GATEWAY else TC_LABEL_AND
            log_messages = [f" {doc_name} / {gateway_type} ".center(50, '-')]
            filtered_gateways = []
            for i, (sentence_gateways, sentence_token_predictions) in enumerate(zip(gateways, predictions)):
                filtered_sentence_gateways = []
                for g in sentence_gateways:
                    if g[2].endswith(gateway_type):
                        # check if keyword extracted gateway is not predicted as gateway in token classification
                        if sentence_token_predictions[g[1]] != gateway_type_token_label:
                            log_msg = f"dropped [sent={i}] {gateway_type} {g} " \
                                      f"-> token mismatch: {sentence_token_predictions[g[1]]}"
                            if self.mode == LOG:
                                filtered_sentence_gateways.append(g)
                            elif self.mode == DROP:
                                pass  # do not add g to filtered list
                        else:
                            log_msg = f"kept [sent={i}] {gateway_type} {g} " \
                                      f"-> token match: {sentence_token_predictions[g[1]]}"
                            filtered_sentence_gateways.append(g)
                        log_messages.append(log_msg)
                filtered_gateways.append(filtered_sentence_gateways)

            # write filter logs into log file (when it exists -> it only exists in when evaluating all documents)
            if self.filtering_log_level == FILE:
                if os.path.isdir(self.results_folder):
                    with open(os.path.join(self.results_folder, f"dropped_{gateway_type}s.txt"), "a") as file:
                        file.writelines(log_messages)
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
                                                         keywords=LITERATURE, output_format=BENCHMARK,
                                                         ensemble_path="/home/japutz/master-thesis/scripts/token_cls/data/logs/GatewayTokenClassifier_train.py-2022-11-19_074241-au=not,bs=8,e=True,e=1,f=2,l=all,olw=0.1,r=ft,ss=og,sg=42,se=0-29,sw=True,us=True",
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