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

from labels import *


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

    @abstractmethod
    def predict_activity_pair(self, doc_name, a1, a2) -> str:
        """
        prediction method to classify an activity pair
        activity format: <TODO>
        :param doc_name: document in which a1 and a2 occur
        :param a1: activity 1
        :param a2: activity 2
        :return: prediction as str label (see label set above)
        """
        pass


class RandomBaselineRelationClassifier(RelationClassifier):
    """
    Dummy classifier baseline that predicts always a random label
    """
    def predict_activity_pair(self, doc_name, a1, a2) -> str:
        return random.choice(label_set)


class DFBaselineRelationClassifier(RelationClassifier):
    """
    Dummy classifier baseline that predicts always a DF relation
    """
    def predict_activity_pair(self, doc_name, a1, a2) -> str:
        return DF



