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
from collections import Counter

from petreader.labels import *

from relation_approaches.RelationClassifier import RelationClassifier, GoldstandardRelationClassifier, \
    RandomBaselineRelationClassifier, DFBaselineRelationClassifier
from PetReader import pet_reader
from labels import *
from utils import GatewayExtractionException, debugging

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
        self.gateway_type_confidence = None
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
            return self.gateway_type in [XOR_GATEWAY, XOR_GATEWAY_OPT]
        elif gateway_type == AND_GATEWAY:
            return self.gateway_type == AND_GATEWAY
        elif gateway_type == NO_GATEWAY_RELATIONS:
            return self.gateway_type == NO_GATEWAY_RELATIONS
        else:
            raise ValueError(f"Only {XOR_GATEWAY}, {AND_GATEWAY} and {NO_GATEWAY_RELATIONS} are allowed for evaluation")

    def __repr__(self):
        return f"Gateway (type={self.gateway_type};confidence={self.gateway_type_confidence})" \
               f"\n    split={self.split_point}\n    merge={self.merge_point}"

    def __str__(self) -> str:
        return f"Gateway (type={self.gateway_type};confidence={self.gateway_type_confidence}) " \
               f"| split={self.split_point} | merge={self.merge_point} | " \
               f"branch_activity_relations={self.branch_activity_relations}"

    def to_json(self) -> Dict:
        return {
            "type": self.gateway_type,
            "confidence": self.gateway_type_confidence,
            "split_point": self.split_point.__repr__(),
            "merge_point": self.merge_point.__repr__(),
            "branch_activity_relations": self.branch_activity_relations
        }


# B) MAIN CLASS


class GatewayExtractor:
    """
    extracts Gateways in a rule-based manner using relations between activities provided by a RelationClassifier
    """

    def __init__(self, relation_classifier: RelationClassifier, full_branch_vote: bool = True):
        """
        defines a new GatewayExtractor by passing the RelationClassifier
        :param relation_classifier: RelationClassifier obj
        :param full_branch_vote: flag if full branches should be used for gateway type determination
        """
        self.relation_classifier = relation_classifier
        self.full_branch_vote = full_branch_vote

    def extract_document_gateways(self, doc_name: str, doc_processing_number: int) -> List[Gateway]:
        """
        extracts the gateways of one document
        :param doc_name: doc name
        :param doc_processing_number: number in current processing
        :return: list of Gateway objects
        """
        logger.info(f" {doc_name} ({doc_processing_number}) ".center(150, "*"))
        doc_activities = pet_reader.get_activities_in_relation_approach_format(doc_name)

        relations = []
        for a1, a2 in itertools.combinations(doc_activities, 2):
            relations.append([a1, a2, self.relation_classifier.predict_activity_pair(doc_name, a1, a2)])

        split_points = self._detect_split_points(relations)
        merge_points = self._detect_merge_points(relations)
        split_points_merged = self._merge_gateway_point_candidates(split_points)
        merge_points_merged = self._merge_gateway_point_candidates(merge_points)
        gateways = self._detect_gateways(doc_name, relations, split_points_merged, merge_points_merged)
        gateways = self._classify_gateways(gateways, relations)
        return gateways

    @debugging
    def extract_document_gateways_debug(self, doc_name: str) -> List[Gateway]:
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
        print("-" * 100)

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

        # 3) Detect gateways
        gateways = self._detect_gateways(doc_name, relations, split_points_merged, merge_points_merged)

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
        relations = self._filter_relations(relations, label=DIRECTLY_FOLLOWING)
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
        relations = self._filter_relations(relations, label=DIRECTLY_FOLLOWING)
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

    def _detect_gateways(self, doc_name: str, relations: List[Tuple], split_points: List[GatewayPoint],
                         merge_points: List[GatewayPoint]) -> List[Gateway]:
        """
        detect gateways in a rule-based way from a set of relations and detected split and merge points
        gateway points are iterated by order of first activity and a merge point get always assigned to the latest
        opened gateway (most close split point)
        :param doc_name: doc name
        :param relations: list of activity relations
        :param split_points: list of gateway split points
        :param merge_points: list of gateway merge points
        :return: list of (unclassified) Gateways
        """
        # assumes that gateway split/merges are mentioned in correct order in text
        gateway_points = split_points + merge_points
        gateway_points.sort(key=lambda p: (p.earliest_activity[0], p.earliest_activity[1],
                                           # assure that SPLIT is preferred in case of optional gateway where one
                                           # activity is part of pointing activities of  split and merge
                                           0 if p.type == SPLIT else 1))

        gateways = []
        logger.debug(" Sequence of Gateway points ".center(100, '-'))
        for p in gateway_points:
            logger.debug(p)
            if p.type == SPLIT:
                gateways.append(Gateway(split_point=p))
            elif p.type == MERGE:
                opened_gateways = [g for g in gateways if g.merge_point is None]
                if opened_gateways:
                    newest_opened_gateway = opened_gateways[-1]
                    newest_opened_gateway.merge_point = p
        logger.info(f"Gateway detection {doc_name} -> detected {len(gateways)} gateways from {len(relations)} relations"
                    f" and {len(split_points)} split and {len(merge_points)} merge points")
        return gateways

    def _setup_next_relations_dict_for_current_doc(self, relations) -> None:
        """
        stores in a dict the list of directly following activities to every activity
        :param relations: relation set as base
        """
        next_relations = {}
        for r in relations:
            if r[2] == DIRECTLY_FOLLOWING:
                if str(r[0]) in next_relations:
                    next_relations[str(r[0])].append(r[1])
                else:
                    next_relations[str(r[0])] = [r[1]]
        self.next_relations_current_doc = next_relations

    def _classify_gateways(self, gateways: List[Gateway], relations: List[Tuple]) -> List[Gateway]:
        """
        classify if a gateway with a defined split (and optionally merge point) is exclusive or parallel
        :param gateways: list of gateways
        :param relations: list of activity relations
        :return: list of gateways enriched with gateway types
        """
        self._setup_next_relations_dict_for_current_doc(relations)
        logger.info(" Gateways ".center(100, '-'))
        for g in gateways:
            branch_start_activities = self.get_branch_start_activities(g)
            if self.full_branch_vote:
                branches = self.get_full_branch(g, branch_start_activities)
            else:
                branches = [[a] for a in branch_start_activities]
            branch_activities_relations = self.get_branch_activity_relations(branches, relations)

            self.determine_gateway_type_from_relations(g, branch_activities_relations)
            g.branch_activity_relations = branch_activities_relations

        for g in gateways:
            logger.info(g.__repr__())

        return gateways

    @staticmethod
    def get_branch_start_activities(g: Gateway) -> List[Tuple]:
        """
        Return the start activity for each branch of a gateway
        :param g: gateway
        :return: list of activities
        """
        branch_activities = g.split_point.receiving_activities
        # filter merge point activities if merge point exists (for empty branches)
        if g.merge_point:
            branch_activities = [a for a in branch_activities if a not in g.merge_point.receiving_activities]
        return branch_activities

    def get_full_branch(self, g: Gateway, branch_start_activities: List[Tuple]) -> List[List[Tuple]]:
        """
        extend branch that contains start activities with activities until merge point
        """
        merge_activities = g.merge_point.receiving_activities if g.merge_point else []
        branches = [[a] + self._lookup_next_activities(a, merge_activities)
                    for a in branch_start_activities]
        return branches

    def _lookup_next_activities(self, activity: Tuple, merge_activities: List[Tuple]) -> List[Tuple]:
        """
        Lookup next activities until a (potentially given) merge point. Search is recursive until merge point or end
        :param activity: activity to start search from
        :param merge_activities: merge activities where search should be stopped (empty if no merge point exist)
        :return:
        """
        if str(activity) in self.next_relations_current_doc:
            result = []
            for a in self.next_relations_current_doc[str(activity)]:
                if a not in merge_activities:
                    # add activity and its successors if not yet in result set
                    if a not in result:
                        result.append(a)
                    next_a_activities = self._lookup_next_activities(a, merge_activities)
                    for aa in next_a_activities:
                        if aa not in result:
                            result.append(aa)
            return result
        else:
            return []

    @staticmethod
    def get_branch_activity_relations(branches: List[List[Tuple]], relations: List[Tuple]) -> List[Tuple]:
        """
        get relations between all pairs of activities in all branches
        :param branches: branches defined as two dim list of activities
        :param relations: doc relations
        :return: list of relations
        """
        # all in combinations necessary in case of >2 branches
        branch_activities_relations = []
        for branchA, branchB in itertools.combinations(branches, 2):
            for a1, a2 in itertools.product(*[branchA, branchB]):
                if a1 != a2:
                    try:
                        relation = list(filter(lambda r: (r[0] == a1 and r[1] == a2) or (r[1] == a1 and r[0] == a2),
                                               relations))[0]
                    except IndexError:
                        raise GatewayExtractionException(f"relation of {a1} and {a2} is not relation set")
                    branch_activities_relations.append(relation)
        return branch_activities_relations

    @staticmethod
    def determine_gateway_type_from_relations(g: Gateway, branch_activities_relations: List[Tuple]) -> None:
        """
        Determine gateway type from relations by voting the gateway type by a majority vote from relation labels between
        all activity pairs from all branches (may be limited to start activities if self.full_branch_vote)
        Note: in voting only exclusive/concurrent relations as they indicate possible gateways are included whether in
              the confidence score all relations are included
        """
        if branch_activities_relations:
            relevant_relation_labels = [r[2] for r in branch_activities_relations if r[2] in [EXCLUSIVE, CONCURRENT]]
            if relevant_relation_labels:
                most_common_labels = Counter(relevant_relation_labels).most_common()
                most_common_label = most_common_labels[0]
                g.gateway_type = XOR_GATEWAY if most_common_label[0] == EXCLUSIVE else AND_GATEWAY
                g.gateway_type_confidence = most_common_label[1] / len(branch_activities_relations)
            # no exclusive or concurrent relations are present
            else:
                g.gateway_type = NO_GATEWAY_RELATIONS
                g.gateway_type_confidence = -1
                # logger.warning(f"No gateway relations detected for {g}")
        # special case with only one optional branch -> no relations to activities from other branches
        else:
            g.gateway_type = XOR_GATEWAY_OPT
            g.gateway_type_confidence = 1.0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    gateway_extractor = GatewayExtractor(relation_classifier=GoldstandardRelationClassifier(),
                                         full_branch_vote=True)

    # test one
    gateway_extractor.extract_document_gateways_debug(doc_name="doc-5.3")

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
