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
import itertools
from typing import List, Tuple, Dict

from petreader.labels import *

from labels import *
from PetReader import pet_reader
from utils import GatewayExtractionException, ROOT_DIR, load_pickle, save_as_pickle


logger = logging.getLogger('Data Generation [Activity Relations]')
doc_black_list = ['doc-6.4']


# DEBUGGING VERSION: notebooks/relation_approaches/activity_relation_data_preparation.ipynb

# A) HELPER METHODS


def _transform_relations(relations: List[Dict]) -> List[Dict]:
    """
    transform list of relations into an internal representation
    :param relations: list of dicts
    :return: list of dicts with two tuples each
    """
    results = [{SOURCE: (r[SOURCE_SENTENCE_ID], r[SOURCE_HEAD_TOKEN_ID], r[SOURCE_ENTITY], r[SOURCE_ENTITY_TYPE]),
                TARGET: (r[TARGET_SENTENCE_ID], r[TARGET_HEAD_TOKEN_ID], r[TARGET_ENTITY], r[TARGET_ENTITY_TYPE])}
               for r in relations]
    return results


def _unique_ordered_flows(flows: List[Dict]) -> List[Dict]:
    """
    make list of flows unique and order by sentence/word index of source
    :param flows: list of flows
    :return: unique and ordered list
    """
    flows_u = []
    for f in flows:
        if f not in flows_u:
            flows_u.append(f)
    flows_u.sort(key=lambda f: (f[SOURCE][0], f[SOURCE][1]))
    return flows_u


def _get_linked_entities(gateway: Tuple, flow_relations: List[Dict]):
    """
    get all entities that are linked to a given gateway via a flow relation
    :param gateway: gateway
    :param flow_relations: set of flows to check
    :return: list of entities that are linked to gateway
    """
    return [r[TARGET] for r in flow_relations if r[SOURCE] == gateway]


def _get_linked_entities_via_condition(gateway: Tuple, flow_relations: List[Dict]) -> List[Tuple]:
    """
    get all entities that are linked to a given gateway indirectly via a condition specifications
    :param gateway: gateway
    :param flow_relations: set of flows to check
    :return: list of entities that are linked to gateway indirect via a condition
    """
    linked_entities = []
    for r in flow_relations:
        if r[SOURCE] == gateway and r[TARGET][3] == CONDITION_SPECIFICATION:
            targets = [r2[TARGET] for r2 in flow_relations if r2[SOURCE] == r[TARGET]]
            linked_entities.extend(targets)
    return linked_entities
    # return [[r2[TARGET] for r2 in flow_relations if r2[SOURCE] == r[TARGET]][0]
    #         for r in flow_relations if r[SOURCE] == gateway and r[TARGET][3] == CONDITION_SPECIFICATION]


def _get_sg_gateways(gateway: Tuple, sg_relations: List[Dict]) -> List[Dict]:
    """
    search for gateways that are related to the given gateway via a same gateway relation
    search is conducted recursively to support multi branch gateways (>2 branches)
    :param gateway: gateway
    :param sg_relations: set of same gateway relations
    :return: list of gateways that are linked to gateway via a same gateway relation
    """
    results = []
    for sg in sg_relations:
        if sg[SOURCE] == gateway:
            results.append(sg[TARGET])
            recursive_gateways = _get_sg_gateways(sg[TARGET], sg_relations)
            if recursive_gateways:
                results.extend(recursive_gateways)
    return results


def _get_following_flows_by_text_structure(element: Tuple, flow_relations: List[Dict]) -> List[Dict]:
    """
    Return the flows that follow a element later in the text
    :param element: start element
    :param flow_relations: set of flows to check
    :return: following flows
    """
    return [f for f in flow_relations if f[SOURCE][0] > element[0]
            or (f[SOURCE][0] == element[0] and f[SOURCE][1] >= element[1])]


def _get_previous_flows_by_text_structure(element: Tuple, flow_relations: List[Dict]) -> List[Dict]:
    """
    Return the flows that preceed a element later in the text
    :param element: start element
    :param flow_relations: set of flows to check
    :return: previous flows
    """
    return [f for f in flow_relations if f[SOURCE][0] < element[0]
            or (f[SOURCE][0] == element[0] and f[SOURCE][1] <= element[1])]


def _get_number_incoming_flows(element: Tuple, flow_relations: List[Dict]) -> int:
    """
    return number of incoming flows of an element
    """
    return len([f for f in flow_relations if f[TARGET] == element])


def _get_number_outgoing_flows(element: Tuple, flow_relations: List[Dict]) -> int:
    """
    return number of outgoing flows of an element
    """
    return len([f for f in flow_relations if f[SOURCE] == element])


def _get_merge_point_search_flows(element: Tuple, flow_relations: List[Dict]) -> List[Dict]:
    """
    return list of flows which are searched in course of merge point search from given element on
    :param element: element to search merge point for
    :param flow_relations: set of flows to check
    :return: list of filtered flows
    """
    # start with flows following by text structure
    following_flows = _get_following_flows_by_text_structure(element, flow_relations)

    # check for other flows of the target element before because order in text can differ from process logic order
    additional_flows = []
    for f in following_flows:
        additional_flows.extend(_get_following_flows_by_text_structure(f[TARGET], flow_relations))

    # add flows ongoing from directly linked activities because structure in text can be different then process
    # structure (e.g. doc-1.1 parallel gateway)
    directly_linked_entities = _get_linked_entities(element, flow_relations)
    for e in directly_linked_entities:
        additional_flows.extend(_get_following_flows_by_text_structure(e, flow_relations))

    return _unique_ordered_flows(following_flows + additional_flows)


def _find_next_merge_point(element: Tuple, flow_relations: List[Dict]) -> Tuple:
    """
    find the merge point of a given (gateway) element, i.e. find next activity that has multiple incoming flows
    :param element: element to search merge point for
    :param flow_relations: set of flows to check
    :return: merge point entity
    """
    relevant_flows = _get_merge_point_search_flows(element, flow_relations)
    next_targets = []
    unclosed_gateways = 1
    for f in relevant_flows:
        # another gateway opened that has to be closed first
        # check for incoming flows == 1 because with > 1 gateway is merge point as well
        if f[TARGET][3] in [XOR_GATEWAY, AND_GATEWAY] and _get_number_incoming_flows(f[TARGET], flow_relations) == 1:
            unclosed_gateways += 1
        if f[TARGET] in next_targets:
            # one closing found
            unclosed_gateways -= 1
            # check if all opened gateways are closed
            if unclosed_gateways == 0:
                return f[TARGET]
        else:
            next_targets.append(f[TARGET])
    return None


def _get_following_flows(element: Tuple, flow_relations: List[Dict]) -> List[Dict]:
    """
    get following flows of a given element -> followings by text structure and repeat for them the procedure
    -> necessary because text structure provides not always the full set in case there is another link before
    :param element: element to search merge point for
    :param flow_relations: set of flows to check
    :return: list of flows
    """
    # start with flows following by text structure
    following_flows = _get_following_flows_by_text_structure(element, flow_relations)

    # check for other links to the element before the element itself
    for f in flow_relations:
        if f[SOURCE] == element:
            following_flows.extend(_get_following_flows_by_text_structure(f[SOURCE], flow_relations))

    return _unique_ordered_flows(following_flows)


def _get_activities_until_merge_point(element, next_merge, flow_relations):
    """
    return all activities between given element and next given merge point based on flow relations/connections
    if merge point is None, return all activities until the end
    """
    activities_between = [element]

    # iterate twice because semantical structure does not always follows textual structure -> in first run not all are
    # captured
    # duplicates will be created, but filtered after again
    def dummy():
        for f in flow_relations:
            # if source of new flow is in already recorded elements and (no merge exist or target is before merge)
            if f[SOURCE] in activities_between \
                    and (not next_merge or
                         (f[TARGET][0] < next_merge[0] or (
                                 f[TARGET][0] == next_merge[0] and f[TARGET][1] < next_merge[1]))):
                activities_between.append(f[TARGET])

    dummy()
    # remove start element
    activities_between = activities_between[1:]
    dummy()

    # make unique again
    activities_between_u = []
    for a in activities_between:
        if a not in activities_between_u:
            activities_between_u.append(a)

    return activities_between


def _get_last_activities(flow, flow_relations):
    """
    search for last (transitively) linked activities (recursively) before current flow
    :param flow: flow to start reversed search for
    :param flow_relations: set of flows
    :return: list of transitive connected activities
    """
    last_activities = []
    relevant_flows = _get_previous_flows_by_text_structure(flow[SOURCE], flow_relations)
    last_element = flow[SOURCE]

    # search from this flow to search reversed for last activities
    while not last_activities:
        source_flows = [f for f in relevant_flows if f[TARGET] == last_element]
        temp_new_activities = []
        for source_flow in source_flows:
            # a) base case -> activity found
            if source_flow[SOURCE][3] == ACTIVITY:
                temp_new_activities.append(source_flow[SOURCE])
            # b) recursive case -> continue search from flow before
            else:
                temp_new_activities.extend(_get_last_activities(source_flow, relevant_flows))
        last_activities.extend(temp_new_activities)

    return last_activities


def _enrich_doc_start_flow(flow_relations):
    """
    add an additional flow relation for the document start
    :param flow_relations: normal set of flow relations
    :return: extended set of flow relations
    """
    new_first_flow = {SOURCE: (-1, -1, None, DOC_START), TARGET: flow_relations[0][SOURCE]}
    return [new_first_flow] + flow_relations


def _filter_merge_point(merge_point: Tuple, entity_list: List[Tuple]) -> List[Tuple]:
    """
    filter a given merge point from a given list
    :param merge_point: merge point list
    :param entity_list: list
    :return: filtered list
    """
    return [e for e in entity_list if e != merge_point]


def _filter_cond_spec(entity_list: List[Tuple]) -> List[Tuple]:
    """
    filter condition specifications from a given entity list
    :param entity_list: entities
    :return: filtered entities
    """
    return [e for e in entity_list if e[3] != CONDITION_SPECIFICATION]


def _create_branch_relations(doc_name: str, flow_relations: List[Dict], gateway: Tuple, merge_point: Tuple,
                             branch_start_entities: List[Tuple]) -> List[Tuple]:
    """
    create all relations between all entities of all combinations of branches
    :param doc_name: doc name
    :param flow_relations: set of flow relations
    :param gateway: gateway of branches
    :param merge_point: gateways merge point
    :param branch_start_entities: the start entities of every gateway branch
    :return: list of relations in tuple format
    """
    relations = []
    # add exclusive/concurrent relations between (multiple) activities of branches
    # first create list of activities for each branch
    # exclude merge point because its not part of exclusive/concurrent relations
    activity_branches = ([[e] + (_get_activities_until_merge_point(e, merge_point, flow_relations))
                          for e in _filter_merge_point(merge_point, branch_start_entities)])

    # second create connections between all activities of each pair of branches
    for branchA, branchB in itertools.combinations(activity_branches, 2):
        for e1, e2 in itertools.product(*[branchA, branchB]):
            if e1[3] == ACTIVITY and e2[3] == ACTIVITY:  # omit gateways or condition specs
                relations.append((doc_name, e1, e2,
                                  EXCLUSIVE if gateway[3] == XOR_GATEWAY else CONCURRENT, "branches"))

    return relations


# B) MAIN METHOD


def generated_activity_relations(doc_names: List[str] = None) -> List[Tuple]:
    """
    generate activity relation data
    relations are represented as (doc_name, (a1), (a2), relation type, comment)
    split/merge points are represented as directly follow relations from activity before/after the gateway and
    first/last activities inside the gateway
    :param doc_names: list of documents which should be processed, if None -> all
    :return: list of relations in format
    """

    # prepare cache path where to load/save data
    cache_path = os.path.join(ROOT_DIR, f"data/other/data_cache/activity_relation_data_"
                                        f"[{str(doc_names) if doc_names else 'all'}]")
    # reload from cache if already exists
    if os.path.exists(cache_path):
        relations = load_pickle(cache_path)
        logger.info(f"Reloaded activity relation data ({len(relations)}) from cache")
        return relations

    # if not generate data and save in cache
    relations = []
    for i, doc_name in enumerate(pet_reader.document_names):

        if i % 5 == 0:
            logger.info(f"Processed {i} documents")

        if (doc_names and doc_name not in doc_names) or doc_name in doc_black_list:
            continue

        # 1) Search for relations using gateways
        doc_relations = pet_reader.relations_dataset.GetRelations(pet_reader.get_document_number(doc_name))
        flow_relations = _transform_relations(doc_relations[FLOW])
        flow_relations = _enrich_doc_start_flow(flow_relations)
        same_gateway_relations = _transform_relations(doc_relations[SAME_GATEWAY])

        # special case: remove last flow for doc-9.5 because this is a whole process loop
        if doc_name == 'doc-9.5':
            flow_relations = flow_relations[:-1]

        for f in flow_relations:

            # a) DIRECTLY FOLLOWING RELATIONS
            if f[SOURCE][3] == f[TARGET][3] == ACTIVITY:
                relations.append((doc_name, f[SOURCE], f[TARGET], DF, "normal df"))

            # b) RELATIONS INVOLVING GATEWAYS
            if f[TARGET][3] in [XOR_GATEWAY, AND_GATEWAY]:

                # extract source activity of current flow for pairing with following activities of gateway (f[TARGET])
                if f[SOURCE][3] == ACTIVITY:
                    source_activities = [f[SOURCE]]
                # if gateways are nested/referring each other -> lookup previous last normal activity recursively
                elif f[SOURCE][3] in [CONDITION_SPECIFICATION, XOR_GATEWAY, AND_GATEWAY]:
                    source_activities = _get_last_activities(f, flow_relations)
                # if gateway is at document start, no flows from connected activities of gateway to any source
                # activities can be created
                elif f[SOURCE][3] == DOC_START:
                    source_activities = []
                else:
                    raise GatewayExtractionException("Other flow combination!")

                gateway = f[TARGET]
                gateway_merge_point = _find_next_merge_point(gateway, flow_relations)
                branch_start_entities = []

                # create flows from possible multiple incomes to current gateway (only in case of directly nested
                # gateways) to possible multiple outcomes (normal for gateways)

                # - 1) in case of direct entity (activity or further gateway) link without conditon and same gateway
                # cases: exlusive 'or' gateways || parallel gateways
                directly_linked_entities = _get_linked_entities(gateway, flow_relations)
                # add relations of activities before to (directly linked) gateway activities via DF
                for e in directly_linked_entities:
                    if e[3] == ACTIVITY:
                        for source_activity in source_activities:
                            relations.append((doc_name, source_activity, e, DF, "g -> a"))
                directly_linked_entities_filtered = _filter_merge_point(gateway_merge_point, directly_linked_entities)
                directly_linked_entities_filtered = _filter_cond_spec(directly_linked_entities_filtered)
                branch_start_entities.extend(directly_linked_entities_filtered)

                # - 2) in case of indirect link via condition specification or same gateway relations
                gateway_branches_entities_directly_linked = []
                condition_spec_linked = _get_linked_entities_via_condition(gateway, flow_relations)
                for e in condition_spec_linked:
                    if e[3] == ACTIVITY:
                        for source_activity in source_activities:
                            relations.append((doc_name, source_activity, e, DF, "g -> cond -> a"))
                        gateway_branches_entities_directly_linked.append(e)
                    # not activity is linked, but other (gateway) from which fol. act. will be included as well
                    else:
                        gateway_branches_entities_directly_linked.append(e)
                branch_start_entities.extend(gateway_branches_entities_directly_linked)

                # - 3) detect same gateways and repeat procedure for them
                sg_entities_linked = []
                sg_gateways = _get_sg_gateways(gateway, same_gateway_relations)
                for sg_gateway in sg_gateways:
                    # directly linked
                    sg_linked_entities = _get_linked_entities(sg_gateway, flow_relations)
                    for e in sg_linked_entities:
                        if e[3] == ACTIVITY:
                            for source_activity in source_activities:
                                relations.append((doc_name, source_activity, e, DF, "g -> sg -> a"))
                            sg_entities_linked.append(e)
                        # not activity is linked, but other (gateway) from which fol. act. will be included as well
                        elif e[3] in [XOR_GATEWAY, AND_GATEWAY]:
                            sg_entities_linked.append(e)
                    # linked via condition
                    sg_gateway_condition_spec_linked = _get_linked_entities_via_condition(sg_gateway, flow_relations)
                    for e in sg_gateway_condition_spec_linked:
                        if e[3] == ACTIVITY:
                            for source_activity in source_activities:
                                relations.append((doc_name, source_activity, e, DF, "g -> sg -> cond -> a"))
                            sg_entities_linked.append(e)
                        # not activity is linked, but other (gateway) from which fol. act. will be included as well
                        else:
                            sg_entities_linked.append(e)
                branch_start_entities.extend(sg_entities_linked)

                # Create relations between activities of all branches
                relations.extend(_create_branch_relations(doc_name, flow_relations, gateway, gateway_merge_point,
                                                          branch_start_entities))

    # filter duplicates & sort
    relations_final = []
    for r in relations:
        if r not in relations_final and r[1] != r[2]:
            relations_final.append(r)
    # sort by doc, g1 sentence idx, g1 word idx, g2 sentence idx, g2 word idx
    relations_final.sort(key=lambda r: (r[0], r[1][0], r[1][1], r[2][0], r[2][1]))

    # save in cache
    save_as_pickle(relations_final, cache_path)
    logger.info(f"Saved {len(relations_final)} to cache")

    return relations_final


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    relations = generated_activity_relations()
