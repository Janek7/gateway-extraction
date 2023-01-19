# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import logging
from abc import ABC, abstractmethod
import random
from typing import Tuple, List, Dict
import itertools

from PetReader import pet_reader
from labels import *
from relation_approaches.activity_relation_data_preparation import get_activity_relations
from utils import GatewayExtractionException

"""
label set: (imported from labels)
DF = 'directly_following'
EXCLUSIVE = 'exclusive'
CONCURRENT = 'concurrent'
NON_RELATED = 'non_related'
"""
label_set = [DF, EXCLUSIVE, CONCURRENT, NON_RELATED]

logger = logging.getLogger('Relation Classifier')


# A) RelationClassifier classes


class RelationClassifier(ABC):
    """
    abstract base class for RelationClassifiers
    """

    def __init__(self):
        logger.info(f"Initialize a {self.__class__.__name__}")

    @abstractmethod
    def predict_activity_pair(self, doc_name: str, activity_1: Tuple, activity_2: Tuple) -> str:
        """
        prediction method to classify an activity pair
        activity format: tuple of (sentence idx, word idx, token_list, 'ACTIVITY')
        :param doc_name: document in which a1 and a2 occur
        :param activity_1: activity 1
        :param activity_2: activity 2
        :return: prediction as str label (see label set above)
        """
        pass


class RandomBaselineRelationClassifier(RelationClassifier):
    """
    Dummy classifier baseline that predicts always a random label
    """

    def predict_activity_pair(self, doc_name, activity_1, activity_2) -> str:
        return random.choice(label_set)


class DFBaselineRelationClassifier(RelationClassifier):
    """
    Dummy classifier baseline that predicts always a DF relation
    """

    def predict_activity_pair(self, doc_name, activity_1, activity_2) -> str:
        return DF


class GoldstandardRelationClassifier(RelationClassifier):
    """
    Classifier that returns always gold standard relation
    """

    def __init__(self):
        super().__init__()
        self.relation_data = get_activity_relations(return_type=dict)

    def predict_activity_pair(self, doc_name, activity_1, activity_2) -> str:
        """
        predict relation of a given pair of activities in a document
        :param doc_name: document
        :param activity_1:
        :param activity_2:
        :return: relation label (see head of script)
        """
        target_relation = list(filter(lambda r: r[DOC_NAME] == doc_name and
                                                ((r[ACTIVITY_1] == activity_1 and r[ACTIVITY_2] == activity_2) or
                                                 (r[ACTIVITY_1] == activity_2 and r[ACTIVITY_2] == activity_1)),
                                      self.relation_data))
        if len(target_relation) > 1:
            raise GatewayExtractionException(f"Multiple relations of {activity_1} and {activity_2} found")
        if target_relation:
            return target_relation[0][RELATION_TYPE]
        else:
            raise GatewayExtractionException(f"No relation of {activity_1} and {activity_2} found")


# B) APPLICATION OF RelationClassifier


def classify_documents(relation_classifier: RelationClassifier, doc_names: List[str] = None) -> Dict:
    """
    evaluate a RelationClassifier by applying it to given list of doc_names (if empty, evaluate all)
    :param relation_classifier: RelationClassifier instance
    :param doc_names: documents to evaluate
    :return:
    """
    relations = {}
    if not doc_names:
        doc_names = pet_reader.document_names
    logger.info(f"Create activity relation predictions for {len(doc_names)} documents using a "
                f"{relation_classifier.__class__.__name__}")

    # Create predictions for every activity pair in every document in given list
    for i, doc_name in enumerate(doc_names):

        if i % 5 == 0:
            logger.info(f"Processed {i} documents")

        doc_relations = []
        doc_activities = pet_reader.get_activities_in_relation_approach_format(doc_name)
        for a1, a2 in itertools.combinations(doc_activities, 2):
            doc_relations.append((a1, a2, relation_classifier.predict_activity_pair(doc_name, a1, a2)))

        relations[doc_name] = doc_relations

    return relations


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    goldstandard_classifier = GoldstandardRelationClassifier()
    relations = classify_documents(goldstandard_classifier, ["doc-9.5"])

    for doc_name, doc_relations in relations.items():
        print(doc_name.center(100, '-'))
        for r in doc_relations:
            print(r)
