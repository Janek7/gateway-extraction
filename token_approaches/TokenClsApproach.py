#!/usr/bin/env python3

import logging
from typing import List, Tuple

from petreader.labels import *

from GatewayTokenClassifier import GatewayTokenClassifier, convert_predictions_into_labels
from Ensemble import Ensemble
from KeywordsApproach import KeywordsApproach
from PetReader import pet_reader
from token_approaches.token_data_preparation import preprocess_tokenization_data
from utils import config, set_seeds
from labels import *

logger = logging.getLogger('Token Classification Approach')


class TokenClsApproach(KeywordsApproach):
    """
    TODO: rewrite class structure for appropriate inheritance
    use token classification results for gateway extraction -> do not use keyword extraction
    extend to use rules of key word approach for flow extraction, but filtering here means take only and all predictions
    from token classification model
    """

    def __init__(self, approach_name: str = None, include_and: bool = True, output_format: str = BENCHMARK,
                 same_xor_gateway_threshold: int = 1, output_folder: str = None,
                 # class specific params:
                 ensemble_path: str = None):
        """
        :param approach_name: description of approach to use in result folder name; if not set use key word variant
        :param include_and: flag if AND gateway predictions should be included
        :param same_xor_gateway_threshold: threshold to recognize subsequent (contradictory xor) gateways as same
        :param output_format: output format of extracted element and flows; available: benchmark, pet
        :param output_folder: name of output folder; if none -> create based on approach name
        :param ensemble_path: path of ensemble model to restore weights from;
                              if None, a random initialized model will be used
        """
        # pass here LITERATURE as dummy value for keywords -> no effect, because gateways will be overwritten
        super().__init__(approach_name=approach_name, keywords=LITERATURE, output_format=output_format,
                         same_xor_gateway_threshold=same_xor_gateway_threshold, output_folder=output_folder)
        self.include_and = include_and
        if ensemble_path:
            self.token_classifier = Ensemble(model_class=GatewayTokenClassifier, ensemble_path=ensemble_path)
        else:  # only for debugging
            self.token_classifier = GatewayTokenClassifier(args=None)
        set_seeds(config[SEED], "Reset after initialization of GatewayTokenClassifierEnsemble")

    def filter_gateways(self, doc_name: str, xor_gateways: List[List[Tuple[str, int, str]]],
                        and_gateways: List[List[Tuple[str, int, str]]]) \
            -> Tuple[List[List[Tuple[str, int, str]]], List[List[Tuple[str, int, str]]]]:
        """
        filtering in this approach means take only the gateways from the token classifier
        :param doc_name: name of document to which gateways belongs
        :param xor_gateways: list of xor gateways of target document in PET format (NOT USED HERE)
        :param and_gateways: list of and gateways of target document in PET format (NOT USED HERE)
        :return: xor gateways, and gateways (same format, but all taken from gateway token classifier)
        """
        # preprocess data
        tokens, _, _, word_ids = preprocess_tokenization_data(sample_numbers=pet_reader.get_doc_sample_ids(doc_name))
        original_tokens = pet_reader.get_doc_sentences(doc_name)

        # predict token labels with GatewayTokenClassifier
        predictions = self.token_classifier.predict(tokens)
        predictions = convert_predictions_into_labels(predictions, word_ids)

        # transform into PET format
        xor_gateways = []
        and_gateways = []
        for token_predictions, tokens in zip(predictions, original_tokens):
            xor_gateways_sentence = []
            and_gateways_sentence = []
            for i, (token, token_label) in enumerate(zip(tokens, token_predictions)):
                if token_label == TC_LABEL_XOR:
                    xor_gateways_sentence.append((token, i, f"B-{XOR_GATEWAY}"))
                elif token_label == TC_LABEL_AND and self.include_and:
                    and_gateways_sentence.append((token, i, f"B-{AND_GATEWAY}"))
            xor_gateways.append(xor_gateways_sentence)
            and_gateways.append(and_gateways_sentence)

        return xor_gateways, and_gateways


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    set_seeds(config[SEED], "Set first seed")
    keyword_filtered_approach = TokenClsApproach(approach_name='token_cls_og_syn_incl_and', include_and=True,
                                                 output_format=BENCHMARK,
                                                 ensemble_path="/home/japutz/master-thesis/scripts/token_cls/data/logs/GatewayTokenClassifier_train.py-2022-11-19_074241-au=not,bs=8,e=True,e=1,f=2,l=all,olw=0.1,r=ft,ss=og,sg=42,se=0-29,sw=True,us=True")
    if True:
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
