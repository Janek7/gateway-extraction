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
from abc import ABC, abstractmethod
import random
from typing import Tuple

from labels import *
from relation_approaches.activity_relation_data_preparation import get_activity_relations

"""
label set: (imported from labels)
DF = 'directly_following'
EXCLUSIVE = 'exclusive'
CONCURRENT = 'concurrent'
NON_RELATED = 'non_related'
"""
label_set = [DF, EXCLUSIVE, CONCURRENT, NON_RELATED]

logger = logging.getLogger('Relation Classifier')


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
        target_relation = list(filter(lambda r: r[DOC_NAME] == doc_name and r[ACTIVITY_1] == activity_1
                                                and r[ACTIVITY_2] == activity_2,
                                      self.relation_data))
        if target_relation:
            return target_relation[0][RELATION_TYPE]
        # TODO: remove after non_related is introduced as well -> throw exception here then
        else:
            return NON_RELATED


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    goldstandard_classifier = GoldstandardRelationClassifier()
    print(goldstandard_classifier.predict_activity_pair('doc-2.2',
                                                        (11, 27, ['resolved'], 'Activity'),
                                                        (13, 10, ['sent', 'out'], 'Activity')))
