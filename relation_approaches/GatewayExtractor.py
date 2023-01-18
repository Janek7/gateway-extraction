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
import itertools

from petreader.labels import *

from relation_approaches.RelationClassifier import RelationClassifier, GoldstandardRelationClassifier
from PetReader import pet_reader
from labels import *
from utils import GatewayExtractionException

logger = logging.getLogger('GatewayExtractor')

# internal constants
SPLIT = 'split'
MERGE = 'merge'


class GatewayPoint:

    def __init__(self, type, pointing_activities, receiving_activities):
        self.type = type
        self.pointing_activities = pointing_activities
        self.receiving_activities = receiving_activities

    def __repr__(self):
        return f"Gateway {self.type} [pointing activities={self.pointing_activities};" \
               f"receiving activities={self.receiving_activities}]"


class GatewayExtractor:

    def __init__(self, relation_classifier: RelationClassifier):
        self.relation_classifier = relation_classifier

    def extract_document_gateways(self, doc_name: str) -> List:
        """

        :param doc_name:
        :return:
        """
        doc_activities = pet_reader.get_activities_in_relation_approach_format(doc_name)

        # 1) Create relations using relation classifier
        relations = []
        for a1, a2 in itertools.combinations(doc_activities, 2):
            relations.append((a1, a2, self.relation_classifier.predict_activity_pair(doc_name, a1, a2)))
        relations_existing = self._filter_relations(relations, exclude_label=NON_RELATED)

        for r in relations_existing:
            print(r)
        print("-"*100)

        # 2) detect split and merge points in relation set
        print("split_points")
        split_points = self._detect_split_points(relations)
        for sp in split_points:
            print(sp)
        print("-" * 100)
        print("split points merged")
        split_points_merged = self._merge_gateway_point_candidates(split_points)
        for sp in split_points_merged:
            print(sp)
        print("-" * 100)

        print("merge points")
        merge_points = self._detect_merge_points(relations)
        for mp in merge_points:
            print(mp)
        print("-" * 100)
        print("merge points merged")
        merge_points_merged = self._merge_gateway_point_candidates(merge_points)
        for mp in merge_points_merged:
            print(mp)
        print("-" * 100)

        gateways = []
        return gateways

    def _detect_split_points(self, relations: List[Tuple]) -> List[GatewayPoint]:
        """
        detects points (split/merge) in relation set, i.e. activities with multiple outgoing/incoming flows or
        directly following relations
        :param relations: relation set
        :return: list of activities
        """
        relations = self._filter_relations(relations, label=DF)
        # count outgoing flows for every activity
        gateway_candidates = []
        for r in relations:
            tmp = list(filter(lambda gc: r[0] in gc.pointing_activities, gateway_candidates))
            if tmp:
                tmp[0].receiving_activities.append(r[1])
            else:
                gateway_candidates.append(GatewayPoint(SPLIT, [r[0]], [r[1]]))
        return [gc for gc in gateway_candidates if len(gc.receiving_activities) > 1]

    def _detect_merge_points(self, relations: List[Tuple]) -> List[GatewayPoint]:
        """
        detects points (split/merge) in relation set, i.e. activities with multiple outgoing/incoming flows or
        directly following relations
        :param relations: relation set
        :return: list of activities
        """
        relations = self._filter_relations(relations, label=DF)
        # count outgoing flows for every activity
        gateway_candidates = []
        for r in relations:
            tmp = list(filter(lambda gc: r[1] in gc.receiving_activities, gateway_candidates))
            if tmp:
                tmp[0].pointing_activities.append(r[0])
            else:
                gateway_candidates.append(GatewayPoint(MERGE, [r[0]], [r[1]]))
        return [gc for gc in gateway_candidates if len(gc.pointing_activities) > 1]

    @staticmethod
    def _merge_gateway_point_candidates(gateway_candidates):
        final_candidates = []
        if gateway_candidates:
            key_attr = 'pointing_activities' if gateway_candidates[0].type == MERGE else 'receiving_activities'
            merge_attr = 'receiving_activities' if gateway_candidates[0].type == MERGE else 'pointing_activities'
            for gc in gateway_candidates:
                tmp = list(filter(lambda fc: getattr(fc, key_attr) == getattr(gc, key_attr), final_candidates))
                if tmp:
                    getattr(tmp[0], merge_attr).extend(getattr(gc, merge_attr))
                else:
                    final_candidates.append(gc)
        return final_candidates

    @staticmethod
    def _filter_relations(relations, label=None, exclude_label=None):
        """
        filters relation list on given label
        :param relations: relations
        :param exclude_label: label to exclude
        :return: filtered list
        """
        if (label and exclude_label) or (not label and not exclude_label):
            raise GatewayExtractionException
        elif label:
            return list(filter(lambda r: r[2] == label, relations))
        elif exclude_label:
            return list(filter(lambda r: r[2] != exclude_label, relations))


if __name__ == '__main__':
    gateway_extractor = GatewayExtractor(relation_classifier=GoldstandardRelationClassifier())
    # simple one
    gateway_extractor.extract_document_gateways(doc_name="doc-3.8")

    # optional gateway at the end; multiple closing behind each other
    # gateway_extractor.extract_document_gateways(doc_name="doc-1.1")

    # includes gateway at start
    # gateway_extractor.extract_document_gateways(doc_name="doc-10.14")

    # includes gateways directly following -> split point is other gateways merge point
    # gateway_extractor.extract_document_gateways(doc_name="doc-3.2")
