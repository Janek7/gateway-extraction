# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

from typing import Dict, List
from copy import deepcopy
import logging
import json
from abc import abstractmethod, ABC

import pandas as pd
from petreader.labels import *

from relation_approaches.AbstractClassificationBenchmark import AbstractClassificationBenchmark
from relation_approaches.GatewayExtractor import GatewayExtractor, Gateway
from relation_approaches.RelationClassifier import DFBaselineRelationClassifier, GoldstandardRelationClassifier
from relation_approaches.activity_relation_data_preparation import DOC_BLACK_LIST
from relation_approaches import metrics
from utils import ROOT_DIR, save_as_pickle, flatten_list
from PetReader import pet_reader
from labels import *

logger = logging.getLogger('Gateway Extraction Benchmark')
GATEWAY_TYPES = [XOR_GATEWAY, AND_GATEWAY, NO_GATEWAY_RELATIONS]


class GatewayExtractionBenchmark(AbstractClassificationBenchmark):
    """
    Creates and evaluates extraction of gateways created by a GatewayExtractor instance
    """

    def __init__(self, approach_name: str, gateway_extractor: GatewayExtractor, output_folder: str = None,
                 round_digits: int = 2):
        self.gateway_extractor = gateway_extractor

        # prepare output folder
        if not output_folder:
            output_folder = os.path.join(ROOT_DIR,
                                         f"data/results_relation_approaches/gateway_extraction/{approach_name}")

        AbstractClassificationBenchmark.__init__(self, GATEWAY_TYPES, approach_name, output_folder, round_digits)

    def evaluate_documents(self, doc_names: List[str] = None):
        """
        evaluate list of documents with relation_classifier
        :param doc_names: doc names, if none -> all
        :return:
        """
        if not doc_names:
            doc_names = pet_reader.document_names
            doc_names = [d for d in doc_names if d not in DOC_BLACK_LIST]
        logger.info(f"Create predictions for {len(doc_names)} documents")

        gateway_extractions = {doc_name: self.gateway_extractor.extract_document_gateways(doc_name, i+1)
                               for i, doc_name in enumerate(doc_names)}
        logger.info(f"Compute metrics of all labels in all documents")
        all_doc_metrics = self.compute_document_label_metrics(gateway_extractions)
        logger.info(f"Average label metrics")
        label_avg_metrics = self.average_label_wise(all_doc_metrics)
        logger.info(f"Average metrics over all documents")
        overall_avg_metrics = metrics.average_metrics([m for label, m in label_avg_metrics.items()], self.round_digits)

        logger.info(f"Write results & predictions to {self.output_folder}")
        self.write_results(gateway_extractions)
        self.write_metrics(all_doc_metrics, label_avg_metrics, overall_avg_metrics)

    def compute_document_label_metrics(self, gateway_extractions: Dict[str, List[Gateway]]) -> Dict[str, Dict]:
        """
        Compute metrics per gateway label/type and document
        :param gateway_extractions: dictionary with extracted gateways per document
        :return: dictionary with structure {doc-name: {label: {metric: value}}}
        """
        all_doc_metrics = {}
        for i, doc_name in enumerate(gateway_extractions.keys()):
            logger.info(f"Compute metrics for {doc_name} ({i+1}/{len(gateway_extractions.keys())})")
            doc_metrics = {label: self.evaluate_gateway_extractions(doc_name, gateway_extractions[doc_name],
                                                                    label=label) for label in self.labels}
            all_doc_metrics[doc_name] = doc_metrics
        return all_doc_metrics

    @abstractmethod
    def evaluate_gateway_extractions(self, doc_name, gateway_extractions: List[Gateway], label: str) -> Dict:
        """
        compute precision, recall and f1 score for given extracted gateways of a document
        :param doc_name: document name
        :param gateway_extractions: extracted gateways as a list of Gateway objects
        :param label: filter gateway extractions and gold standard to this label
        :return: dictionary with metrics
        """
        pass

    def write_results(self, gateway_extractions: Dict[str, List[Gateway]]) -> None:
        """
        write single extractions to pickle, json and txt file
        :param gateway_extractions: dictionary of gateway extractions document-wise
        :return:
        """
        save_as_pickle(gateway_extractions, os.path.join(self.output_folder, "predictions.pkl"))

        with open(os.path.join(self.output_folder, "predictions.json"), 'w') as file:
            json.dump({doc_name: [g.to_json() for g in gateways] for doc_name, gateways in gateway_extractions.items()},
                      file, indent=4)

        with open(os.path.join(self.output_folder, "predictions.txt"), 'w') as file:
            for doc_name, gateways in gateway_extractions.items():
                file.write(f" {doc_name} ".center(100, '-') + "\n")
                for g in gateways:
                    file.write(str(g) + "\n")
                file.write("\n" * 3)


class SimpleGatewayTypeAndNumberBenchmark(GatewayExtractionBenchmark):
    """
    Evaluates extraction of gateways created by a GatewayExtractor instance by just checking a gateway of this type
    is contained in the PET gold standard
    Notes:
        - gateway information as location or tokens are not available when extracting with 'GatewayExtractor' based on
          activity pair relations
        - gateways that are related with a same gateway relation in PET are already extracted as one gateway
          i.e. in PET exist more gateways -> handle by subtracting number of same gateway relations
    """

    def evaluate_gateway_extractions(self, doc_name, gateway_extractions: List[Gateway], label: str) -> Dict:
        """
        compute precision, recall and f1 score for given extracted gateways of a document
        :param doc_name: document name
        :param gateway_extractions: extracted gateways as a list of Gateway objects
        :param label: filter gateway extractions and gold standard to this label
        :return: dictionary with metrics
        """
        pred_gateways = [g for g in deepcopy(gateway_extractions) if g.check_type_for_evaluation(label)]

        same_gateway_relations = pet_reader.get_doc_relations(doc_name)[SAME_GATEWAY]
        same_gateway_relations_filtered = [sgr for sgr in same_gateway_relations if sgr[SOURCE_ENTITY_TYPE] == label]
        if label == XOR_GATEWAY:
            gold_gateways = flatten_list(pet_reader.token_dataset.GetXORGateways(doc_name))
        elif label == AND_GATEWAY:
            gold_gateways = flatten_list(pet_reader.token_dataset.GetANDGateways(doc_name))
        elif label == NO_GATEWAY_RELATIONS:
            gold_gateways = []
        else:
            raise ValueError(f"'{label}' is not a valid Gateway type")

        number_gold_gateways = len(gold_gateways)
        number_gold_gateways -= len(same_gateway_relations_filtered)
        number_pred_gateways = len(pred_gateways)

        if number_pred_gateways > number_gold_gateways:
            tp = number_gold_gateways
            remaining_preds = number_pred_gateways - number_gold_gateways
        elif number_pred_gateways <= number_gold_gateways:
            tp = number_pred_gateways
            remaining_preds = number_gold_gateways - number_pred_gateways

        # fp = number of elements in predictions that remain unmatched
        fp = remaining_preds
        # fn = number of elements in gold standard that remain unmatched
        fn = number_gold_gateways - tp

        return self.compute_metrics_dict(tp, fp, fn, number_gold_gateways)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    geb = SimpleGatewayTypeAndNumberBenchmark(approach_name="ge=standard_rc=goldstandard_limited",
                                              gateway_extractor=GatewayExtractor(GoldstandardRelationClassifier()))
    # evaluate all documents
    geb.evaluate_documents(["doc-1.1", "doc-1.2"])

    # evaluate single documents
    # geb.evaluate_documents(["doc-1.1", "doc-1.2"])
