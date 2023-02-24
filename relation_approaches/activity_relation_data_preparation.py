# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import logging
import json
import itertools
from typing import List, Tuple, Dict
import random

from petreader.labels import *

from labels import *
from PetReader import pet_reader
from utils import  GatewayExtractionException, ROOT_DIR, load_pickle, save_as_pickle, load_loop_flows, \
    load_activity_relation_black_list_docs

logger = logging.getLogger('Data Generation [Activity Relations]')
# load predefined black list docs here once for the whole activity relation module
DOC_BLACK_LIST = load_activity_relation_black_list_docs()
loop_flows = load_loop_flows()

# lists for stats counting of nested gateways & branch lengths
nested_gateways = []
branch_lengths = []


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

    flows = _unique_ordered_flows(following_flows + additional_flows)
    flows_filtered = []
    for f in flows:
        if f[TARGET] != element:
            flows_filtered.append(f)

    return flows_filtered


def _find_next_merge_point(doc_name: str, element: Tuple, flow_relations: List[Dict]) -> Tuple:
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
            tmp = {DOC_NAME: doc_name, "parent": element, "nested_gateway": f[TARGET]}
            if tmp not in nested_gateways:
                nested_gateways.append(tmp)
        if f[TARGET] in next_targets:
            # one closing found
            unclosed_gateways -= 1
            # check if all opened gateways are closed
            if unclosed_gateways == 0:
                return f[TARGET]
        else:
            next_targets.append(f[TARGET])
    return None


def _get_following_flows(element: Tuple, flow_relations: List[Dict], same_gateway_relations: List[Dict]) -> List[Dict]:
    """
    get following flows of a given element -> followings by text structure and repeat for them the procedure
    -> necessary because text structure provides not always the full set in case there is another link before
    :param element: element to search merge point for
    :param flow_relations: set of flows to check
    :param same_gateway_relations: set of same gateway relations to check
    :return: list of flows
    """
    # start with flows following by text structure
    following_flows = _get_following_flows_by_text_structure(element, flow_relations)

    for f in following_flows:
        # check for links to same gateways and their following activities/branches
        if f[TARGET][3] in [XOR_GATEWAY, AND_GATEWAY]:
            sg_gateways = _get_sg_gateways(f[TARGET], same_gateway_relations)
            for same_gateway in sg_gateways:
                sg_following_flows = _get_following_flows(same_gateway, flow_relations, same_gateway_relations)
                following_flows.extend(sg_following_flows)

    return _unique_ordered_flows(following_flows)


def _get_entities_until_merge_point(element: Tuple, next_merge: Tuple, flow_relations: List[Dict],
                                    same_gateway_relations: List[Dict]) -> List[Tuple]:
    """
    return all activities between given element and next given merge point based on flow relations/connections
    if merge point is None, return all activities until the end
    :param element: element to search merge point for
    :param next_merge: next merge point until which entities should be returned
    :param flow_relations: set of flows to check
    :param same_gateway_relations: set of same gateway relations to check
    :return: list of flows
    """
    entities_between = [element]

    # iterate twice because semantical structure does not always follows textual structure AND because of same gateways
    # -> in first run maybe not all are captured
    # duplicates will be created, but filtered after again
    def dummy():
        for f in flow_relations:
            # if source of new flow is in already recorded elements and (no merge exist or target is before merge)
            if f[SOURCE] in entities_between \
                    and (not next_merge or
                         (f[TARGET][0] < next_merge[0] or (
                                 f[TARGET][0] == next_merge[0] and f[TARGET][1] < next_merge[1]))):
                entities_between.append(f[TARGET])

                if f[TARGET][3] in [XOR_GATEWAY, AND_GATEWAY]:
                    entities_between.extend(_get_sg_gateways(f[TARGET], same_gateway_relations))

    dummy()
    # remove start element
    entities_between = entities_between[1:]
    dummy()

    # make unique again
    entities_between_u = []
    for a in entities_between:
        if a not in entities_between_u:
            entities_between_u.append(a)

    return entities_between


def _get_last_activities(flow: Dict, flow_relations: List[Dict]):
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


def _enrich_doc_start_flow(flow_relations: List[Dict]) -> List[Dict]:
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


def log_branch_lengths(doc_name: str, activity_branches: List[List[Tuple]]) -> None:
    """
    Log lengths of branches to shared list
    :param doc_name: doc_name
    :param activity_branches: list of branches (each a list of entities)
    :return:
    """
    gateway_branch_lengths = []
    for b in activity_branches:
        unique_activities = []
        for a in b:
            if a[3] == ACTIVITY and a not in unique_activities:
                unique_activities.append(a)
        gateway_branch_lengths.append(len(unique_activities))
    branch_lengths.extend([{DOC_NAME: doc_name, "branch_length": length} for length in gateway_branch_lengths])


def _create_branch_relations(doc_name: str, flow_relations: List[Dict], same_gateway_relations: List[Dict],
                             gateway: Tuple, merge_point: Tuple, branch_start_entities: List[Tuple]) -> List[Tuple]:
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
    activity_branches = ([[e] + (_get_entities_until_merge_point(e, merge_point, flow_relations,
                                                                 same_gateway_relations))
                          for e in _filter_merge_point(merge_point, branch_start_entities)])
    log_branch_lengths(doc_name, activity_branches)

    # second create connections between all activities of each pair of branches
    for branchA, branchB in itertools.combinations(activity_branches, 2):
        for e1, e2 in itertools.product(*[branchA, branchB]):
            if e1[3] == ACTIVITY and e2[3] == ACTIVITY:  # omit gateways or condition specs
                relations.append((doc_name, e1, e2,
                                  EXCLUSIVE if gateway[3] == XOR_GATEWAY else CONCURRENT, "branches"))

    return relations


def _create_remaining_relations(doc_name: str, normal_relations: List[Tuple], doc_flow_relations: List[Dict],
                                doc_same_gateway_relations: List[Dict]) -> List[Tuple]:
    """
    create relations between all pairs that do not have a relation assigned yet
    add a following relation if there is a path between activities; if not exclusive as fall back
    :param doc_name: document
    :param normal_relations: set of already extracted relations (df, exclusive, concurrent)
    :return: list of non-related relations
    """
    # get all activities of a document and transfer to internal format
    doc_activities = pet_reader.get_activities_in_relation_approach_format(doc_name)

    # create relations between all pairs that do not have a relation assigned yet
    new_relations = []
    for a1, a2 in itertools.combinations(doc_activities, 2):
        normal_relations_filtered = list(
            filter(lambda r: (r[1] == a1 and r[2] == a2) or (r[2] == a1 and r[1] == a2), normal_relations))
        if not normal_relations_filtered:
            try:
                if _find_path_to_target(doc_flow_relations, doc_same_gateway_relations, a1, a2):
                    new_relations.append((doc_name, a1, a2, EVENTUALLY_FOLLOWING, "not directly following"))
                else:
                    new_relations.append((doc_name, a1, a2, EXCLUSIVE, "fallback - no path found"))
            # is only activated in case of last four activities at the end of doc-2.2 in # parallel gateway
            except RecursionError:
                new_relations.append((doc_name, a1, a2, EVENTUALLY_FOLLOWING, "not directly following"))
    return new_relations


def _find_path_to_target(doc_flow_relations: List[Dict], doc_same_gateway_relations: List[Dict], entity: Tuple,
                         target_activity: Tuple) -> bool:
    """
    check if there is a path via sequence flows between two activities
    method is applied recursively
    :param doc_flow_relations: flow relations of whole document
    :param doc_same_gateway_relations: same gateway relations of whole document
    :param entity: start entity
    :param target_activity: target activity
    :return: false/true if path (not) exists
    """
    entity_flows = list(filter(lambda f: f["source"] == entity, doc_flow_relations))
    connected_entities = [f["target"] for f in entity_flows]

    for gateway in [e for e in connected_entities if e[3] in [XOR_GATEWAY, AND_GATEWAY]]:
        connected_entities.extend(_get_same_gateways(gateway, doc_same_gateway_relations))

    if target_activity in connected_entities:
        return True
    else:
        return any([_find_path_to_target(doc_flow_relations, doc_same_gateway_relations, e, target_activity)
                    for e in connected_entities])


def _get_same_gateways(gateway: Tuple, same_gateway_relations: List[Dict]) -> List[Tuple]:
    """
    search recursively for all gateways that are related via a same gateway relation to the given gateway
    recursive search necessary because of pair-wise transitive connections in same gateway relations
    :param gateway: gateway
    :param same_gateway_relations: same gateway relations of whole document
    :return: list of gateways
    """
    same_gateways = [sg["target"] for sg in same_gateway_relations if sg["source"] == gateway]
    for same_gateway in same_gateways:
        same_gateways.extend(_get_same_gateways(same_gateway, same_gateway_relations))
    return same_gateways


def _relations_to_dict(relations: List[Tuple]) -> List[Dict]:
    """
    transforms every relation into a dict format
    :param relations: relations as tuples
    :return: relations as dicts
    """
    return [{DOC_NAME: r[0], ACTIVITY_1: r[1], ACTIVITY_2: r[2], RELATION_TYPE: r[3], COMMENT: r[4]} for r in relations]


# B) MAIN METHOD


def _compute_activity_relations(doc_names: List[str] = None, drop_loops: bool = True, return_type: type = List,
                                overwrite: bool = False) -> List[Tuple]:
    """
    generate activity relation data
    relations are represented as (doc_name, (a1), (a2), relation type, comment)
    split/merge points are represented as directly follow relations from activity before/after the gateway and
    first/last activities inside the gateway
    :param doc_names: list of documents which should be processed, if None -> all
    :param drop_loops: flag if flow connections that cause loops should be dropped
    :param return_type: type of single relation
    :param overwrite: flag if data should be generated new and overwrite an already existing cache
    :return: list of relations with data [doc_name, (a1), (a2), relation type, comment] in list or dict form
    """

    # prepare cache path where to load/save data
    cache_path = os.path.join(ROOT_DIR, f"data/other/data_cache/activity_relation/relations_[drop_loops={drop_loops}]"
                                        f"[{str(doc_names) if doc_names else 'all'}]")
    # reload from cache if already exists
    if os.path.exists(cache_path) and not overwrite:
        relations = load_pickle(cache_path)
        logger.info(f"Reloaded activity relation data [drop_loops={drop_loops}] from cache ({len(relations)})")
        if return_type == dict:
            relations = _relations_to_dict(relations)
        return relations

    # if not generate data and save in cache
    relations = []
    for i, doc_name in enumerate(pet_reader.document_names):
        if i % 5 == 0:
            logger.info(f"Processed {i} documents")

        doc_relations = []

        if (doc_names and doc_name not in doc_names) or doc_name in DOC_BLACK_LIST:
            continue

        # 1) Search for relations using gateways
        pet_relations = pet_reader.relations_dataset.GetRelations(pet_reader.get_document_number(doc_name))
        flow_relations = _transform_relations(pet_relations[FLOW])
        flow_relations = _enrich_doc_start_flow(flow_relations)
        same_gateway_relations = _transform_relations(pet_relations[SAME_GATEWAY])

        if drop_loops and doc_name in loop_flows:
            flow_relations.remove(loop_flows[doc_name])

        for f in flow_relations:

            # a) DIRECTLY FOLLOWING RELATIONS
            if f[SOURCE][3] == f[TARGET][3] == ACTIVITY:
                doc_relations.append((doc_name, f[SOURCE], f[TARGET], DIRECTLY_FOLLOWING, "normal df"))

            # b) RELATIONS INVOLVING GATEWAYS
            elif f[TARGET][3] in [XOR_GATEWAY, AND_GATEWAY]:

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
                gateway_merge_point = _find_next_merge_point(doc_name, gateway, flow_relations)
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
                            doc_relations.append((doc_name, source_activity, e, DIRECTLY_FOLLOWING, "g -> a"))
                directly_linked_entities_filtered = _filter_merge_point(gateway_merge_point, directly_linked_entities)
                directly_linked_entities_filtered = _filter_cond_spec(directly_linked_entities_filtered)
                branch_start_entities.extend(directly_linked_entities_filtered)

                # - 2) in case of indirect link via condition specification or same gateway relations
                gateway_branches_entities_directly_linked = []
                condition_spec_linked = _get_linked_entities_via_condition(gateway, flow_relations)
                for e in condition_spec_linked:
                    if e[3] == ACTIVITY:
                        for source_activity in source_activities:
                            doc_relations.append((doc_name, source_activity, e, DIRECTLY_FOLLOWING, "g -> cond -> a"))
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
                                doc_relations.append((doc_name, source_activity, e, DIRECTLY_FOLLOWING, "g -> sg -> a"))
                            sg_entities_linked.append(e)
                        # not activity is linked, but other (gateway) from which fol. act. will be included as well
                        elif e[3] in [XOR_GATEWAY, AND_GATEWAY]:
                            sg_entities_linked.append(e)
                    # linked via condition
                    try:
                        sg_gateway_condition_spec_linked = _get_linked_entities_via_condition(sg_gateway,
                                                                                              flow_relations)
                        for e in sg_gateway_condition_spec_linked:
                            if e[3] == ACTIVITY:
                                for source_activity in source_activities:
                                    doc_relations.append(
                                        (doc_name, source_activity, e, DIRECTLY_FOLLOWING, "g -> sg -> cond -> a"))
                                sg_entities_linked.append(e)
                            # not activity is linked, but other (gateway) from which fol. act. will be included as well
                            else:
                                sg_entities_linked.append(e)
                    except IndexError:
                        # in case of doc-2.1 were relation is removed due to loop -> just skip this branch
                        pass
                branch_start_entities.extend(sg_entities_linked)

                # Create relations between activities of all branches
                doc_relations.extend(_create_branch_relations(doc_name, flow_relations, same_gateway_relations,
                                                              gateway, gateway_merge_point, branch_start_entities))

        doc_relations.extend(_create_remaining_relations(doc_name, doc_relations, flow_relations,
                                                         same_gateway_relations))

        # add doc_relations to global set of relations
        relations.extend(doc_relations)

    # filter duplicates & sort
    relations_final = []
    for r in relations:
        # check if pair (order of gateways doesnt matter) is already in set
        if r not in relations_final and (r[0], r[2], r[1], r[3], r[4]) not in relations_final and r[1] != r[2]:
            relations_final.append(r)
    relations_final.sort(key=lambda r: (r[0], r[1][0], r[1][1], r[2][0], r[2][1]))

    # save in cache
    save_as_pickle(relations_final, cache_path)
    logger.info(f"Saved {len(relations_final)} to cache [drop_loops={drop_loops}]")

    # log more detailed information
    with open(os.path.join(ROOT_DIR, "data/paper_stats/activity_relation/nested_gateways.json"), 'w') as file:
        nested_gateways.sort(key=lambda g: g[DOC_NAME])
        json.dump({"nested_gateways": nested_gateways}, file, indent=4)

    with open(os.path.join(ROOT_DIR, "data/paper_stats/activity_relation/branch_lengths.json"), 'w') as file:
        branch_lengths.sort(key=lambda g: g[DOC_NAME])
        json.dump({"branch_lengths": branch_lengths}, file, indent=4)

    if return_type == dict:
        relations_final = _relations_to_dict(relations_final)

    return relations_final


def get_activity_relations(doc_names: List[str] = None, drop_loops: bool = True, return_type: type = List,
                           overwrite: bool = False) -> List[Tuple]:
    """
    return activity relations; see __compute_activity_relations for param descriptions
    IMPORTANT: make sure a seed is set before (for shuffling)
    :return: (filtered) list of relations
    """
    relations = _compute_activity_relations(doc_names, drop_loops, return_type, overwrite)
    # shuffle to drop relations from different documents
    random.shuffle(relations)
    return relations

    # new_relations = []
    # if down_sample_ef:
    #     ef_count = 0
    #     for r in relations:
    #         relation_type = r[RELATION_TYPE] if return_type == dict else r[3]
    #         if relation_type == EVENTUALLY_FOLLOWING:
    #             if ef_count < config[ACTIVITY_RELATION_CLASSIFIER][EVENTUALLY_FOLLOWS_SAMPLE_LIMIT]:
    #                 new_relations.append(r)
    #                 ef_count += 1
    #         else:
    #             new_relations.append(r)
    #     return new_relations
    # else:
    #     return relations


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    relations = get_activity_relations(down_sample_ef=False, doc_names=["doc-2.2"])#, doc_names=["doc-3.6"])
    relations.sort(key=lambda r: (r[0][0], r[0][1], r[1][0], r[1][1]))
    for r in relations:
        print(r[1:4])
    print(len(relations))

    all_relations = get_activity_relations(down_sample_ef=False)
    print(len(all_relations))
    doc_relations = [r for r in all_relations if r[0] == 'doc-2.2']
    print(len(doc_relations))
