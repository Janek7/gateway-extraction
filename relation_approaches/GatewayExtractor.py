# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)


import logging
from typing import List, Tuple, Dict
import itertools

from petreader.labels import *

from relation_approaches.RelationClassifier import RelationClassifier, GoldstandardRelationClassifier
from PetReader import pet_reader
from labels import *
from utils import GatewayExtractionException

logger = logging.getLogger('GatewayExtractor')


# A) HELPER CLASSES


class GatewayPoint:
    """
    Class that represents a gateway split or merge point by defining activities pointing to this point and receiving
    activities that are pointed to
    """

    def __init__(self, type, pointing_activities, receiving_activities):
        """
        create a GatewayPoint
        :param type: gateway point type -> 'split' or 'merge'
        :param pointing_activities: activities pointing to this point
        :param receiving_activities: receiving activities that are pointed to
        """
        self.type = type
        self.pointing_activities = pointing_activities
        self.receiving_activities = receiving_activities

    @property
    def earliest_activity(self):
        """
        :return: return in the text mentioned earliest activity
        """
        pointing_activities = self.pointing_activities.copy()
        pointing_activities.sort(key=lambda a: (a[0], a[1]))
        return pointing_activities[0]

    def __repr__(self):
        return f"Gateway {self.type} [pointing activities={self.pointing_activities};" \
               f"receiving activities={self.receiving_activities}]"

    def __str__(self):
        return self.__repr__()


class Gateway:
    """
    Gateway class that wraps important information of one gateway
    """

    def __init__(self, split_point: GatewayPoint):
        """
        gateway is defined by split point, merge point may be added later optionally
        gateway type and relations of the activities in the branches are added later
        :param split_point: split point
        """
        self.split_point = split_point
        # merge point and gateway type are not known yet
        self.merge_point = None
        self.gateway_type = None
        self.branch_activity_relations = None

    def check_type_for_evaluation(self, gateway_type) -> bool:
        """
        check if the gateway is XOR_GATEWAY or AND_GATEWAY
        method necessary, because GatewayExtractor is able to extract refined exclusive XOR optional gateways (which
        count for evaluation as XOR_GATEWAY as well)
        :param gateway_type: gateway type to check for (XOR_GATEWAY or AND_GATEWAY)
        :return: true/false
        """
        if gateway_type == XOR_GATEWAY:
            return self.gateway_type in [XOR_GATEWAY, XOR_OPT]
        elif gateway_type == AND_GATEWAY:
            return self.gateway_type == AND_GATEWAY
        else:
            raise ValueError(f"Only XOR_GATEAY and AND_GATEWAY are allowed for evaluation")

    def __repr__(self):
        return f"Gateway (type={self.gateway_type})\n    split={self.split_point}\n    merge={self.merge_point}"

    def __str__(self) -> str:
        return f"Gateway (type={self.gateway_type}) | split={self.split_point} | merge={self.merge_point} | " \
               f"branch_activity_relations={self.branch_activity_relations}"

    def to_json(self) -> Dict:
        return {
            "type": self.gateway_type,
            "split_point": self.split_point.__repr__(),
            "merge_point": self.merge_point.__repr__(),
            "branch_activity_relations": self.branch_activity_relations
        }


# B) MAIN CLASS


class GatewayExtractor:
    """
    extracts Gateways in a rule-based manner using relations between activities provided by a RelationClassifier
    """

    def __init__(self, relation_classifier: RelationClassifier):
        """
        defines a new GatewayExtractor by passing the RelationClassifier
        :param relation_classifier: RelationClassifier obj
        """
        self.relation_classifier = relation_classifier

    def extract_document_gateways(self, doc_name: str) -> List[Gateway]:
        """
        extracts the gateways of one document
        :param doc_name: doc name
        :return: list of Gateway objects
        """
        doc_activities = pet_reader.get_activities_in_relation_approach_format(doc_name)

        # 1) Create relations using relation classifier
        relations = []
        for a1, a2 in itertools.combinations(doc_activities, 2):
            relations.append((a1, a2, self.relation_classifier.predict_activity_pair(doc_name, a1, a2)))
        # relations = self._filter_relations(relations, exclude_label=NON_RELATED)

        for r in relations:
            print(r)
        print("-"*100)

        # 2) detect split and merge points in relation set
        # print("split_points")
        split_points = self._detect_split_points(relations)
        # for sp in split_points:
        #     print(sp)
        # print("-" * 100)
        print("split points merged")
        split_points_merged = self._merge_gateway_point_candidates(split_points)
        for sp in split_points_merged:
            print(sp)
        print("-" * 100)

        # print("merge points")
        merge_points = self._detect_merge_points(relations)
        # for mp in merge_points:
        #     print(mp)
        # print("-" * 100)
        # print("merge points merged")
        merge_points_merged = self._merge_gateway_point_candidates(merge_points)
        for mp in merge_points_merged:
            print(mp)
        print("-" * 100)

        # 3) Detect gateways
        gateways = self._detect_gateways(relations, split_points, merge_points)

        # 4) Classify gateways
        gateways = self._classify_gateways(gateways, relations)

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
    def _merge_gateway_point_candidates(gateway_point_candidates):
        """
        merge gateway point candidates:
            merge split point if they are pointing to the same activity(s)
            merge merge point if they are receiving the same activity(s)
        :param gateway_point_candidates: list of GatewayPoint candidates
        :return: merged list of GatewayPoint candidates
        """
        final_candidates = []
        if gateway_point_candidates:
            key_attr = 'pointing_activities' if gateway_point_candidates[0].type == MERGE else 'receiving_activities'
            merge_attr = 'receiving_activities' if gateway_point_candidates[0].type == MERGE else 'pointing_activities'
            for gc in gateway_point_candidates:
                tmp = list(filter(lambda fc: getattr(fc, key_attr) == getattr(gc, key_attr), final_candidates))
                if tmp:
                    getattr(tmp[0], merge_attr).extend(getattr(gc, merge_attr))
                else:
                    final_candidates.append(gc)
        return final_candidates

    @staticmethod
    def _detect_gateways(relations: List[Tuple], split_points: List[GatewayPoint],
                         merge_points: List[GatewayPoint]) -> List[Gateway]:
        """
        detect gateways in a rule-based way from a set of relations and detected split and merge points
        gateway points are iterated by order of first activity and a merge point get always assigned to the latest
        opened gateway (most close split point)
        :param relations: list of activity relations
        :param split_points: list of gateway split points
        :param merge_points: list of gateway merge points
        :return: list of (unclassified) Gateways
        """
        logger.info(f"Detecting gateways from {len(relations)} relations and {len(split_points)} split and "
                    f"{len(merge_points)} merge points")
        # assumes that gateway split/merges are mentioned in correct order in text
        gateway_points = split_points + merge_points
        gateway_points.sort(key=lambda p: (p.earliest_activity[0], p.earliest_activity[1],
                                           # assure that SPLIT is preferred in case of optional gateway where one
                                           # activity is part of pointing activities of  split and merge
                                           0 if p.type == SPLIT else 1))

        gateways = []
        print(" Sequence of Gateway points ".center(100, '-'))
        for p in gateway_points:
            print(p)
            if p.type == SPLIT:
                gateways.append(Gateway(split_point=p))
            elif p.type == MERGE:
                newest_opened_gateway = [g for g in gateways if g.merge_point is None][-1]
                newest_opened_gateway.merge_point = p

        return gateways

    @staticmethod
    def _classify_gateways(gateways: List[Gateway], relations: List[Tuple]) -> List[Gateway]:
        """
        classify if a gateway with a defined split (and optionally merge point) is exclusive or parralel
        :param gateways: list of gateways
        :param relations: list of activity relations
        :return: list of gateways enriched with gateway types
        """
        print(" Gateways ".center(100, '-'))
        for g in gateways:
            branch_activities = g.split_point.receiving_activities

            # all in combinations necessary in case of >2 branches
            branch_activities_relations = []
            for a1, a2 in itertools.combinations(branch_activities, 2):
                try:
                    relation = list(filter(lambda r: (r[0] == a1 and r[1] == a2) or (r[1] == a1 and r[0] == a2),
                                           relations))[0]
                except IndexError:
                    raise GatewayExtractionException(f"relation of {a1} and {a2} is not relation set")
                branch_activities_relations.append(relation)

            # check set of relations -> if all the same, if yes what, or if different ones
            branch_activities_relations_types = [relation[2] for relation in branch_activities_relations]
            branch_activities_relations_types_unique = list(set(branch_activities_relations_types))
            if branch_activities_relations_types_unique == [DF]:
                g.gateway_type = XOR_OPT
            elif branch_activities_relations_types_unique == [EXCLUSIVE]:
                g.gateway_type = XOR_GATEWAY
            elif branch_activities_relations_types_unique == [CONCURRENT]:
                g.gateway_type = AND_GATEWAY
            else:
                g.gateway_type = DIFFERENT_RELATIONS

            g.branch_activity_relations = branch_activities_relations

        for g in gateways:
            print(g.__repr__())

        return gateways

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
    logging.basicConfig(level=logging.DEBUG)
    gateway_extractor = GatewayExtractor(relation_classifier=GoldstandardRelationClassifier())

    # test one
    gateway_extractor.extract_document_gateways(doc_name="doc-9.5")

    # simple one
    # gateway_extractor.extract_document_gateways(doc_name="doc-3.8")

    # not closing gateway at the beginning;optional gateway at the end;nested gateways opening behind each other
    # gateway_extractor.extract_document_gateways(doc_name="doc-1.2")

    # not closing gateway at the beginning;optional gateway at the end;nested gateways closing behind each other
    # gateway_extractor.extract_document_gateways(doc_name="doc-1.1")

    # includes gateway at start
    # gateway_extractor.extract_document_gateways(doc_name="doc-10.14")

    # includes gateways directly following -> split point is other gateways merge point
    # gateway_extractor.extract_document_gateways(doc_name="doc-3.2")
