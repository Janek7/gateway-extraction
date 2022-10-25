from petreader.labels import *
from PetReader import pet_reader
from petbenchmarks.benchmarks import BenchmarkApproach
from petbenchmarks.tokenclassification import TokenClassificationBenchmark
from petbenchmarks.relationsextraction import RelationsExtractionBenchmark
import logging
import json
import os
import shutil
from typing import List, Tuple, Optional, Dict
from labels import *
from utils import format_json_file

logger = logging.getLogger('keyword approach')


class KeywordApproach:

    def __init__(self, approach_name: str = None, keywords: str = LITERATURE, same_xor_gateway_threshold: int = 1,
                 output_format: str = BENCHMARK):
        """
        creates new instance of the basic keyword approach
        :param approach_name: description of approach to use in result folder name; if not set use key word variant
        :param keywords: flag/variant which keywords to use; available: literature, gold, own
        :param same_xor_gateway_threshold: threshold to recognize subsequent (contradictory xor) gateways as same
        :param output_format: output format of extracted element and flows; available: benchmark, pet
        """
        self.approach_name = approach_name
        if not self.approach_name:
            self.approach_name = f"keywords_{keywords}"
        self.keywords = keywords
        self._same_xor_gateway_threshold = same_xor_gateway_threshold
        self.output_format = output_format
        self._xor_keywords = None
        self._and_keywords = None
        self._contradictory_gateways = None
        self._read_and_set_keywords()
        self._read_contradictory_gateways()
        self._processed_doc_gateway_frames = []

        # check string parameters for valid values
        if self.keywords not in [LITERATURE, GOLD, OWN]:
            raise ValueError(f"Key words must be {LITERATURE} or {GOLD}")
        if self.output_format not in [PET, BENCHMARK]:
            raise ValueError(f"Output format must be {PET} or {BENCHMARK}")

    def evaluate_documents(self, doc_names: List[str] = None,
                           tcb: TokenClassificationBenchmark = None, reb: RelationsExtractionBenchmark = None,
                           evaluate_token_cls: bool = True, evaluate_relation_extraction: bool = True) -> None:
        """
        run extraction and evaluation with petbenchmarks
        :param doc_names: list of document names to evaluate, use all as default value
        :param tcb: TokenClassificationBenchmark instance
        :param reb: RelationsExtractionBenchmark instance
        :param evaluate_token_cls: flag to run evaluation of token classification or not
        :param evaluate_relation_extraction: flag to run evaluation of relation extraction or not
        :return: nothing, results are written to .json file
        """
        if not doc_names:
            doc_names = pet_reader.document_names

        # prepare evaluation structures to fill
        if not tcb:
            logger.info("Create TokenClassificationBenchmark ...")
            tcb = TokenClassificationBenchmark()
        process_elements = tcb.GetEmptyPredictionsDict()
        if not reb:
            logger.info("Create RelationsExtractionBenchmark ...")
            reb = RelationsExtractionBenchmark()
        relations = reb.GetEmptyPredictionsDict()

        # process all documents
        logger.info(f"Start processing of {len(doc_names)} documents ...")
        for i, doc_name in enumerate(doc_names):
            if i % 5 == 0:
                logger.info(f"Finished processing of {i} documents.")
            xor_gateways, and_gateways, doc_flows, same_gateway_relations = self.process_document(doc_name)
            process_elements[doc_name][XOR_GATEWAY].extend(xor_gateways)
            process_elements[doc_name][AND_GATEWAY].extend(and_gateways)
            relations[doc_name][FLOW].extend(doc_flows)
            relations[doc_name][SAME_GATEWAY].extend(same_gateway_relations)

        # save results as json
        folder = f'data/results/{self.approach_name}/'
        # clear directory first and then create new
        shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Save results to {folder}")

        process_elements_filename = os.path.join(folder, 'process_elements.json')
        with open(process_elements_filename, 'w') as file:
            json.dump(process_elements, file, indent=4)

        relations_filename = os.path.join(folder, 'relations.json')
        with open(relations_filename, 'w') as file:
            json.dump(relations, file, indent=4)

        # run evaluation
        logger.info(f"Run evaluation")
        if evaluate_token_cls:
            BenchmarkApproach(approach_name=self.approach_name, predictions_file_or_folder=process_elements_filename)
            format_json_file(os.path.join(folder, f"results-{self.approach_name}.json"))
        if evaluate_relation_extraction:
            BenchmarkApproach(approach_name=self.approach_name, predictions_file_or_folder=relations_filename)
            format_json_file(os.path.join(folder, f"results-{self.approach_name}.json"))

    def process_document(self, doc_name: str) -> Tuple[List, List, List, List]:
        """
        extracts and returns gateways and related flow relations for given document
        :param doc_name: document name
        :return: xor_gateways, and_gateways, doc_flows, same_gateway_relations
        """

        # prepare document
        doc_sentences = pet_reader.get_doc_sentences(doc_name)
        doc_activities_enriched = pet_reader.get_index_enriched_activities(doc_name)

        # extract concurrent gateways and related flow relations
        and_gateways = self._extract_gateways(doc_sentences, AND_GATEWAY)
        and_flows = self._extract_concurrent_flows(doc_activities_enriched, and_gateways)

        # extract exclusive gateways and related flow relations
        xor_gateways = self._extract_gateways(doc_sentences, XOR_GATEWAY)
        xor_flows, same_gateway_relations = self._extract_exclusive_flows(doc_activities_enriched, xor_gateways)

        # extract flow relations of gold activities and remove the ones involved in gateway flows
        gold_activity_flows = self._extract_gold_activity_flows(doc_activities_enriched)
        doc_flows = self._merge_flows(gold_activity_flows, xor_flows, and_flows)

        # change format of outputs to BENCHMARK if necessary
        if self.output_format == BENCHMARK:
            # transform gateway entities to simpler benchmark format
            def gateways_to_benchmark(gateways):
                results = []
                for sentence_gateways in gateways:
                    i = 0
                    while i < len(sentence_gateways):
                        gateway_tokens = [sentence_gateways[i][0]]
                        i += 1
                        while i < len(sentence_gateways) and sentence_gateways[i][2].startswith('I-'):
                            gateway_tokens.append(sentence_gateways[i][0])
                            i += 1
                        results.append(gateway_tokens)
                return results
            xor_gateways = gateways_to_benchmark(xor_gateways)
            and_gateways = gateways_to_benchmark(and_gateways)

            # transform relation dictionaries to simpler benchmark format
            relations_to_benchmark = lambda relations: [{SOURCE_ENTITY: r[SOURCE_ENTITY],
                                                         TARGET_ENTITY: r[TARGET_ENTITY]} for r in relations]
            doc_flows = relations_to_benchmark(doc_flows)
            same_gateway_relations = relations_to_benchmark(same_gateway_relations)

        return xor_gateways, and_gateways, doc_flows, same_gateway_relations

    def _extract_gateways(self, sentence_list: List[List[str]], gateway_type: str) -> List[List[Tuple[str, int, str]]]:
        """
        extracts gateways in a key-word-based manner given a document structured in a list of sentences
        if two phrases would match to a token (e.g. 'in the meantime' and 'meantime'), the longer phrase is extracted
        :param sentence_list: document represented as list of sentences (each sentence is a list of tokens)
        :param gateway_type: gateway type to extract ('XOR Gateway' or 'AND Gateway')

        :return: a two dimensional list -> list of tuples (word, position in sentence, tag) for each sentence this
                 produces the same structure as sentences and their tokens and NER labels are annotated in PET dataset
        """
        if gateway_type == XOR_GATEWAY:
            key_words = self._xor_keywords
        elif gateway_type == AND_GATEWAY:
            key_words = self._and_keywords
        else:
            raise ValueError(f"gateway_type must be {XOR_GATEWAY} or {AND_GATEWAY}")
        # sort key words descending by length of words in phrase
        key_words.sort(key=lambda key_word_phrase: len(key_word_phrase.split(" ")), reverse=True)

        # 1) extract gateways
        gateways = []
        for s_idx, tokens in enumerate(sentence_list):
            sentence_gateways = []
            tokens_lower = [t.lower() for t in tokens]
            # create sentence string to search (multi-word) key phrases
            sentence_to_search = f" {' '.join(tokens_lower).lower()} "
            tokens_already_matched_with_key_phrase = []

            # iterate over key phrases
            for key_phrase in key_words:
                key_phrase_to_search = f" {key_phrase} "

                # if key phrase is in sentence, search index and extract
                if key_phrase_to_search in sentence_to_search:
                    key_phrase_tokens = key_phrase.split(" ")

                    # check key phrase for every token
                    for t_idx, token in enumerate(tokens_lower):
                        candidate = True
                        # iterate over key phrase tokens in case of multiple world phrase
                        for key_phrase_token_idx, key_phrase_token in enumerate(key_phrase_tokens):
                            # check if token is not part of key phrase or token is already matched with another phrase
                            # if yes, stop processing candidate
                            if not tokens_lower[t_idx + key_phrase_token_idx] == key_phrase_token or \
                                    t_idx + key_phrase_token_idx in tokens_already_matched_with_key_phrase:
                                candidate = False
                                break

                        # add tokens to result only if all tokens are matched and not already part of a longer phrase
                        if candidate:
                            for i, key_phrase_token in enumerate(key_phrase_tokens):
                                prefix = "B" if i == 0 else "I"
                                # append tuples with extract information as in PET
                                sentence_gateways.append((tokens[t_idx + i], t_idx + i, f"{prefix}-{gateway_type}"))
                                tokens_already_matched_with_key_phrase.append(t_idx + i)

            sentence_gateways.sort(key=lambda gateway_triple: gateway_triple[1])
            gateways.append(sentence_gateways)

        return gateways

    def _preprocess_extracted_gateways(self, extracted_gateways: List[List[Tuple[str, int, str]]], gateway_type: str) \
            -> List[Dict]:
        """
        flatten gateways but keep sentence index; merge multiple gateway tokens into one gateway
        :param extracted_gateways: gateways in PET format (token, index, tag)
        :param gateway_type: type of gateway
        :return: flattened gateway list with entity dictionaries; for format see self._get_entity_dict documentation
        """
        gateways = []
        for sentence_idx, sentence_gateways in enumerate(extracted_gateways):
            sentence_gateways_already_included = []
            for i, gateway in enumerate(sentence_gateways):
                if gateway not in sentence_gateways_already_included:
                    gateway_tokens = [gateway[0]]
                    start_token_idx = gateway[1]
                    # append further tokens of same gateway ('I-' marked)
                    I_index = i + 1
                    while I_index < len(sentence_gateways) and sentence_gateways[I_index][2].startswith('I-'):
                        gateway_tokens.append(sentence_gateways[I_index][0])
                        sentence_gateways_already_included.append(sentence_gateways[I_index])
                        I_index += 1
                    gateway_tokens_lower = [t.lower() for t in gateway_tokens]
                    gateway_tuple = (sentence_idx, start_token_idx, gateway_tokens, gateway_tokens_lower)
                    gateways.append(self._get_entity_dict(gateway_tuple, gateway_type))
        return gateways

    def _extract_gold_activity_flows(self, doc_activity_tokens: List[List[Tuple[str, int]]]) -> List[Dict]:
        """
        Creates simple flows by order of activities
        :param doc_activity_tokens: list of activity tokens (word, idx) for each sentence
        :return: list of flows represented as dicts
        """
        activities_flattened = [(i, activity) for i, sentence_activities in enumerate(doc_activity_tokens)
                                for activity in sentence_activities]
        flow_relations = []
        for i in range(len(activities_flattened) - 1):
            s_idx_1, a1 = activities_flattened[i]
            s_idx_2, a2 = activities_flattened[i + 1]
            a1 = self._get_pet_relation_rep(s_idx_1, a1[1], ACTIVITY, a1[0], source=True)
            a2 = self._get_pet_relation_rep(s_idx_2, a2[1], ACTIVITY, a2[0], source=False)
            flow_relations.append({**a1, **a2})
        return flow_relations

    def _extract_exclusive_flows(self, doc_activity_tokens: List[List[Tuple[str, int]]],
                                 extracted_gateways: List[List[Tuple[str, int, str]]]) -> Tuple[List[Dict], List[Dict]]:
        """
        extracts sequence flows surrounding exclusive gateways based on rules TODO describe rules
        :param doc_activity_tokens: list of activity tokens (word, idx) for each sentence
        :param extracted_gateways: list of own extracted gateway for each sentence
        :return: list of flow relations as source/target dicts; list of same gateway relations as source/target dicts
        """
        sequence_flows = []
        same_gateway_relations = []

        gateways = self._preprocess_extracted_gateways(extracted_gateways, XOR_GATEWAY)
        gateways_involved = []  # list for gateways already involved into sequence flows

        # RULE 1): check for every pair of following gateways if it fits to a gateway constellation with
        # contradictory key words. Gateways must be in range of same_xor_gateway_threshold sentences, otherwise they
        # would be seen as separate ones
        for i in range(len(gateways) - 1):
            g1, g2 = gateways[i], gateways[i + 1]
            # if sentence distances is larger than threshold, reject possible pair
            if abs(g2[ELEMENT][0] - g1[ELEMENT][0]) > self._same_xor_gateway_threshold:
                continue
            # check for every pair of following gateways if it fits to a gateway pair of contradictory key words
            # and check that first gateway is at the beginning of a sentence
            # and check if gateways already matched another pair; possible because of partly same phrase
            for pattern_gateway_1, pattern_gateway_2 in self._contradictory_gateways:
                if g1[ELEMENT][3] == pattern_gateway_1 and g2[ELEMENT][3] == pattern_gateway_2 and g1[ELEMENT][1] == 0 \
                        and g1[ELEMENT] not in gateways_involved and g2[ELEMENT] not in gateways_involved:
                    gateways_involved.append(g1[ELEMENT])
                    gateways_involved.append(g2[ELEMENT])

                    # A) find related activities
                    _, pa_g1, fa_g1, _ = self._get_surrounding_activities(g1, doc_activity_tokens)
                    _, _, fa_g2, ffa_g2 = self._get_surrounding_activities(g2, doc_activity_tokens)

                    # B.1) connect elements to sequence flows
                    # check if fol. activities of g1 and g2 are equal -> if yes, the first branch is without activity
                    empty_branch = fa_g1[ELEMENT] == fa_g2[ELEMENT]
                    # 1) previous activity to first gateway -> split point (if not None because of document start)
                    if pa_g1[ELEMENT]:
                        sequence_flows.append(self._merge_flow(pa_g1, g1))
                    # 2) gateway 1 to following activity and following activity to activity after gateway (second
                    # following of g2) -> merge point
                    # if None because of empty branch then directly there
                    if not empty_branch and fa_g1[ELEMENT]:  # could be None if at document end
                        sequence_flows.append(self._merge_flow(g1, fa_g1))
                        if ffa_g2[ELEMENT]:  # could be None if at document end
                            sequence_flows.append(self._merge_flow(fa_g1, ffa_g2))
                    elif empty_branch and ffa_g2[ELEMENT]:  # could be None if at document end
                        sequence_flows.append(self._merge_flow(g1, ffa_g2))
                    # 3) gateway 2 to following activity and following activity to activity after gateway (second
                    # following of g2) -> merge point
                    if fa_g2[ELEMENT]:  # could be None if at document end
                        sequence_flows.append(self._merge_flow(g2, fa_g2))
                    if ffa_g2[ELEMENT]:  # could be None if at document end
                        sequence_flows.append(self._merge_flow(fa_g2, ffa_g2))

                    # B.2) same gateway flows
                    same_gateway_relations.append(self._merge_flow(g1, g2))

                    # log gateway frame for later usage in flow merging of whole document
                    closing = fa_g2 if fa_g2[ELEMENT] else g2
                    self._log_gateway_frame(g1[ELEMENT][0], g1[ELEMENT][1], g1,
                                            closing[ELEMENT][0], closing[ELEMENT][1], closing)

        # RULE 2): exclusive actions of common pattern "... <activity> ... or ... <activity> ..."
        for g in gateways:
            if g[ELEMENT] not in gateways_involved and g[ELEMENT][3] == ['or']:
                # A) find surrounding activities
                ppa, pa, fa, ffa = self._get_surrounding_activities(g, doc_activity_tokens)

                if pa[ELEMENT] and fa[ELEMENT]:  # check if exist because of document end/start
                    if pa[ELEMENT][0] == g[ELEMENT][0] and fa[ELEMENT][0] == g[ELEMENT][0]:  # check if in same sentence

                        if pa[ELEMENT] is None or fa[ELEMENT] is None:
                            # if not surrounding activities are given, do not wire anything; TODO: maybe drop gateway
                            continue
                        gateways_involved.append(g[ELEMENT])

                        # B) connect elements to sequence flows
                        # 1) second previous activity to gateway -> split point
                        if ppa[ELEMENT]:  # (if not None because of document start)
                            sequence_flows.append(self._merge_flow(ppa, g))
                        # 2) gateway to following activity and previous activity -> exclusive branches
                        sequence_flows.append(self._merge_flow(g, pa))
                        sequence_flows.append(self._merge_flow(g, fa))
                        # 3) exclusive activities to second following activity of gateway -> merge point
                        if ffa[ELEMENT]:  # (if not None because of document end)
                            sequence_flows.append(self._merge_flow(pa, ffa))
                            sequence_flows.append(self._merge_flow(fa, ffa))

                        # log gateway frame for later usage in flow merging of whole document
                        self._log_gateway_frame(pa[ELEMENT][0], pa[ELEMENT][1], g, fa[ELEMENT][0], fa[ELEMENT][1], fa)

        # RULE 3): single-branch gateways: the gateway is related to an activity in the same sentence (order is arbitrary)
        # Assumptiosn: multi-branch gateways are already recognized by rule 1 before; only one activity for the gateway
        for g in gateways:
            if g[ELEMENT] not in gateways_involved and g[ELEMENT][3] != ['or']:
                # A) Prepare elements for flow connections
                ppa, pa, fa, ffa = self._get_surrounding_activities(g, doc_activity_tokens)
                gateways_involved_length_start = len(gateways_involved)

                # B) connect elements to sequence flows -> check which activities exist and how are they located
                # Assumption: only one in the sentence including the gateway
                if fa[ELEMENT] or pa[ELEMENT]:

                    # case 1: no activity before but after in same sentence
                    if fa[ELEMENT] and not pa[ELEMENT] and fa[ELEMENT][0] == g[ELEMENT][0]:
                        sequence_flows.append(self._merge_flow(g, fa))
                        if ffa[ELEMENT]:  # could be None if at document end
                            sequence_flows.append(self._merge_flow(g, ffa))
                            sequence_flows.append(self._merge_flow(fa, ffa))

                        # log gateway frame for later usage in flow merging of whole document
                        self._log_gateway_frame(g[ELEMENT][0], g[ELEMENT][1], g, fa[ELEMENT][0], fa[ELEMENT][1], fa)
                        gateways_involved.append(g[ELEMENT])

                    # case 2: no activity after but before in same sentence
                    elif pa[ELEMENT] and not fa[ELEMENT] and pa[ELEMENT][0] == g[ELEMENT][0]:
                        sequence_flows.append(self._merge_flow(g, pa))
                        # no check for ffa link necessary, because fa is already none
                        # log gateway frame for later usage in flow merging of whole document
                        self._log_gateway_frame(pa[ELEMENT][0], pa[ELEMENT][1], g,
                                                g[ELEMENT][0], g[ELEMENT][1], pa)
                        gateways_involved.append(g[ELEMENT])

                    elif pa[ELEMENT] and fa[ELEMENT]:
                        # case 3: previous is not in the same sentence, but following yes -> activity after gateway
                        if pa[ELEMENT][0] != g[ELEMENT][0] and fa[ELEMENT][0] == g[ELEMENT][0]:
                            # 1) previous activity to gateway -> split point
                            sequence_flows.append(self._merge_flow(pa, g))
                            # 2) gateway to following activity -> exclusive branch
                            sequence_flows.append(self._merge_flow(g, fa))
                            # 3) exclusive activity and gateway to second following activity of gateway -> merge point
                            if ffa[ELEMENT]:  # could be None if at document end
                                sequence_flows.append(self._merge_flow(g, ffa))
                                sequence_flows.append(self._merge_flow(fa, ffa))
                            # log gateway frame for later usage in flow merging of whole document
                            self._log_gateway_frame(g[ELEMENT][0], g[ELEMENT][1], g, fa[ELEMENT][0], fa[ELEMENT][1], fa)
                            gateways_involved.append(g[ELEMENT])

                        # case 4: previous is in the same sentence, but following not -> activity before gateway
                        elif pa[ELEMENT][0] == g[ELEMENT][0] and fa[ELEMENT][0] != g[ELEMENT][0]:
                            # 1) second previous activity to gateway -> split point
                            if ppa[ELEMENT]:  # could be None if at document start
                                sequence_flows.append(self._merge_flow(ppa, g))
                            # 2) gateway to previous activity -> exclusive branch
                            sequence_flows.append(self._merge_flow(g, pa))
                            # 3) exclusive activity and gateway to following activity of gateway -> merge point
                            sequence_flows.append(self._merge_flow(g, fa))
                            sequence_flows.append(self._merge_flow(pa, fa))
                            # log gateway frame for later usage in flow merging of whole document
                            self._log_gateway_frame(pa[ELEMENT][0], pa[ELEMENT][1], g,
                                                    fa[ELEMENT][0], fa[ELEMENT][1], fa)
                            gateways_involved.append(g[ELEMENT])

                if len(gateways_involved) == gateways_involved_length_start:
                    pass  # TODO: remove gateway if no rule for extracting flows could be applied

        return sequence_flows, same_gateway_relations

    def _extract_concurrent_flows(self, doc_activity_tokens: List[List[Tuple[str, int]]],
                                  extracted_gateways: List[List[Tuple[str, int, str]]]) -> List[Dict]:
        """
        extract flow relations for already found AND gateways following the logic:
        - for every gateway, to extract parallel branches, add relation to next activity after and before, because
          that's the pattern how AND key phrases are usually used (oriented by rules of Ferreira et al. 2017)
        - for each case, check over borders if not found in same sentence
        - to extract the flow relation that points to the gateway split point, take the second before
        - to extract the flow relation that points to the gateway merge point, take the second following

        :param doc_activity_tokens: list of activity tokens (word, idx) for each sentence
        :param extracted_gateways: list of own extracted gateway for each sentence
        :return: list of flow relations in source/target dict representation
        """
        relations = []

        for g in self._preprocess_extracted_gateways(extracted_gateways, AND_GATEWAY):
            # 1) Find surrounding activities
            ppa, pa, fa, ffa = self._get_surrounding_activities(g, doc_activity_tokens)

            # 2) Create relations
            # a) flow to gateway: second previous -> gateway
            if ppa[ELEMENT]:  # could be None if at document start
                relations.append(self._merge_flow(ppa, g))
            # b) split into concurrent gateway branches: gateway -> previous; gateway -> following
            # following two None checks (probably) wont never be False, but for safety included
            if pa[ELEMENT]:  # could be None if at document start
                relations.append(self._merge_flow(g, pa))
            if fa[ELEMENT]:  # could be None if at document end
                relations.append(self._merge_flow(g, fa))
            # c) merge branches together: previous -> second following; following -> second following
            if ffa[ELEMENT]:  # could be None if at document end
                relations.append(self._merge_flow(pa, ffa))
                relations.append(self._merge_flow(fa, ffa))

            # log gateway frame for later usage in flow merging of whole document
            self._log_gateway_frame(pa[ELEMENT][0], pa[ELEMENT][1], g,
                                    fa[ELEMENT][0], fa[ELEMENT][1], fa)

        return relations

    def _merge_flows(self, gold_activity_flows: List[Dict], xor_flows: List[Dict], and_flows: List[Dict]) -> List[Dict]:
        """
        merge gold activity flows and flows surrounding gateways into a list of flows for the whole document
        -> gold activity flows are filtered if a gateway flow is involved into an activity
        :param gold_activity_flows: list of flows between gold activities
        :param xor_flows: list of flows involved in XOR gateways
        :param and_flows: list of flows involved in AND gateways
        :return: list of flows describing the whole document
        """
        gateway_flows = xor_flows + and_flows
        logger.info(f"{len(gateway_flows)} gateway flows")
        logger.info(f"{len(gold_activity_flows)} gold activity flows")

        gateway_flows_source_entities = [flow[SOURCE_ENTITY] for flow in gateway_flows]
        # 1) gateway flows as basis
        doc_flows = gateway_flows.copy()

        # 2) check if flow from this entity to another entity exists already in the gateway flows
        # -> prefer flows to gateways than to gold activities
        for flow in gold_activity_flows:
            if not flow[SOURCE_ENTITY] in gateway_flows_source_entities:
                doc_flows.append(flow)
        doc_flows.sort(key=lambda flow: (flow[SOURCE_SENTENCE_ID], flow[SOURCE_HEAD_TOKEN_ID]))

        def get_gateway_frames(flow):
            """
            return a tuple that contains the gateway frames of source and target entity if they are in one, if not None
            :param flow: flow dict in PET format
            :return: tuple of gateway frame indices
            """
            source_entity_gateway_frame = None
            target_entity_gateway_frame = None

            for i, gateway_frame in enumerate(self._processed_doc_gateway_frames):

                if (flow[SOURCE_SENTENCE_ID] == gateway_frame[START_SENTENCE_IDX]
                    and flow[SOURCE_HEAD_TOKEN_ID] >= gateway_frame[START_TOKEN_ID]) or \
                        (flow[SOURCE_SENTENCE_ID] > gateway_frame[START_SENTENCE_IDX]
                         and flow[SOURCE_SENTENCE_ID] < gateway_frame[END_SENTENCE_IDX]) or \
                        (flow[SOURCE_SENTENCE_ID] == gateway_frame[END_SENTENCE_IDX]
                         and flow[SOURCE_HEAD_TOKEN_ID] <= gateway_frame[END_TOKEN_ID]):
                    if not source_entity_gateway_frame:
                        source_entity_gateway_frame = i

                if (flow[TARGET_SENTENCE_ID] == gateway_frame[START_SENTENCE_IDX]
                    and flow[TARGET_HEAD_TOKEN_ID] >= gateway_frame[START_TOKEN_ID]) or \
                        (flow[TARGET_SENTENCE_ID] > gateway_frame[START_SENTENCE_IDX]
                         and flow[TARGET_SENTENCE_ID] < gateway_frame[END_SENTENCE_IDX]) or \
                        (flow[TARGET_SENTENCE_ID] == gateway_frame[END_SENTENCE_IDX]
                         and flow[TARGET_HEAD_TOKEN_ID] <= gateway_frame[END_TOKEN_ID]):
                    if not target_entity_gateway_frame:
                        target_entity_gateway_frame = i

            return source_entity_gateway_frame, target_entity_gateway_frame

        # 3) check if target of this flows is inside a detected gateway frame
        # -> if yes, redirect flow to gateway start / split point
        flows_to_remove = []
        flows_to_add = []
        for flow in doc_flows:
            source_entity_gf, target_entity_gf = get_gateway_frames(flow)
            # check if source and target are not part of the same gateway and the target entity is part of a gateway
            if source_entity_gf != target_entity_gf and target_entity_gf is not None:
                # if the flow target is not the start entity of the gateway (i.e. split point), then rewire
                if flow[TARGET_ENTITY] !=self._processed_doc_gateway_frames[target_entity_gf][START_ENTITY][ELEMENT][2]:
                    flows_to_remove.append(flow)
                    # create new flow between source of current flow and start entity of gateway (split point)
                    flows_to_add.append({**{k: v for k, v in flow.items() if k.startswith("source-")},
                                         **self._processed_doc_gateway_frames[target_entity_gf][START_ENTITY][TARGET]})

        # add/remove new/wrong wired flows after gateway frame check
        doc_flows.extend(flows_to_add)
        for flow in flows_to_remove:
            doc_flows.remove(flow)

        # filtered doc flows for duplicates
        doc_flows_unique = []
        for flow in doc_flows:
            if flow not in doc_flows_unique:
                doc_flows_unique.append(flow)
        doc_flows = doc_flows_unique

        # sort for easier debugging by order in text
        doc_flows.sort(key=lambda flow: (flow[SOURCE_SENTENCE_ID], flow[SOURCE_HEAD_TOKEN_ID]))

        # clear gateway frames of processed doc for next processing
        self._processed_doc_gateway_frames.clear()
        logger.info(f"{len(doc_flows)} doc flows")
        return doc_flows

    def _get_surrounding_activities(self, gateway: 'see _get_entity_dict documentation',
                                    doc_activity_tokens: List[List[Tuple[str, int]]]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        searches for all surrounding activities of a gateway (previous, second previous, following, second following)
        :param gateway: gateway as entity dict
        :param doc_activity_tokens: list of activity lists (describes whole document)
        :return: for activities; each as dictionary
        """
        g = gateway[ELEMENT]
        # 1) get surrounding activities (previous, second previous, following, second following)
        pa = self._get_previous_activity(g[0], g[1], doc_activity_tokens)
        ppa = self._get_previous_activity(g[0], g[1], doc_activity_tokens, skip_first=True)
        fa = self._get_following_activity(g[0], g[1], doc_activity_tokens)
        ffa = self._get_following_activity(g[0], g[1], doc_activity_tokens, skip_first=True)

        return self._get_entity_dict(ppa, ACTIVITY), self._get_entity_dict(pa, ACTIVITY), \
               self._get_entity_dict(fa, ACTIVITY), self._get_entity_dict(ffa, ACTIVITY)

    # HINT: the two following methods follow the same logic, just in different search direction

    def _get_previous_activity(self, sentence_idx: int, token_idx: int,
                               doc_activity_tokens: List[List[Tuple[str, int]]], skip_first: bool = False,
                               one_already_found: bool = False) -> Optional[Tuple[int, int, str]]:
        """
        search recursive for the second last previous activity from a start point defined by sentence_idx and token_idx
        :param sentence_idx: sentence index where to start the search
        :param token_idx: token index where to stat the search
        :param doc_activity_tokens: list of activity lists (describes whole document)
        :param skip_first: True if searching for the second previous activity, False (default) when searching for the previous activity
        :param one_already_found: flag if one activity was already found and skipped for return in course of search for the second previous
        :returns: triple of (sentence idx, token_idx, token) if found, else None
        """
        # search for activities left to the token in target sentence if token is given else in the whole
        if token_idx is not None:
            previous_activities_sentence = [a_t for a_t in doc_activity_tokens[sentence_idx] if a_t[1] < token_idx]
        else:
            previous_activities_sentence = [a_t for a_t in doc_activity_tokens[sentence_idx]]

        # if activities were found, take the last one
        if previous_activities_sentence:
            # return when just searching the first last activity OR when one was already found before
            previous_activity = previous_activities_sentence[-1]
            # 1a) base case: activity found
            if not skip_first or one_already_found:
                return (sentence_idx, previous_activity[1], previous_activity[0])
            # 2a) recursive case: continue search for second previous activity at index of previous activity
            else:
                return self._get_previous_activity(sentence_idx, previous_activity[1], doc_activity_tokens,
                                                   one_already_found=True)
        else:
            next_sentence_idx = sentence_idx - 1
            # 1b) base case: no sentences any more to search
            if next_sentence_idx == -1:
                return None
            # 2b) recursive case: continue search for previous activity in previous sentence
            else:
                return self._get_previous_activity(next_sentence_idx, None, doc_activity_tokens,
                                                   skip_first=skip_first, one_already_found=one_already_found)

    def _get_following_activity(self, sentence_idx: int, token_idx: int,
                                doc_activity_tokens: List[List[Tuple[str, int]]], skip_first: bool = False,
                                one_already_found: bool = False) -> Optional[Tuple[int, int, str]]:
        """
        search recursive for the next following activity from a start point defined by sentence_idx and token_idx
        :param sentence_idx: sentence index where to start the search
        :param token_idx: token index where to stat the search
        :param doc_activity_tokens: list of activity lists (describes whole document)
        :param skip_first: True if searching for the second following activity, False (default) when searching for the following activity
        :param one_already_found: flag if one activity was already found and skipped for return in course of search for the second following
        :returns: triple of (sentence idx, token_idx, token) or None if none was found
        """
        # search for activities right to the token in target sentence if token is given else in the whole
        if token_idx is not None:
            following_activities_sentence = [a_t for a_t in doc_activity_tokens[sentence_idx] if a_t[1] > token_idx]
        else:
            following_activities_sentence = [a_t for a_t in doc_activity_tokens[sentence_idx]]

        # if activities were found, take the first one
        if following_activities_sentence:
            # return when just searching the first following activity OR when one was already found before
            following_activity = following_activities_sentence[0]
            # 1a) base case: activity found
            if not skip_first or one_already_found:
                return (sentence_idx, following_activity[1], following_activity[0])
            # 2a) recursive case: continue search for second following activity at index of following activity
            else:
                return self._get_following_activity(sentence_idx, following_activity[1], doc_activity_tokens,
                                                    one_already_found=True)

        else:
            next_sentence_idx = sentence_idx + 1
            # 1b) base case: no sentences any more to search
            if next_sentence_idx == len(doc_activity_tokens):
                return None
            # 2b) recursive case: continue search for following activity in following sentence
            else:
                return self._get_following_activity(next_sentence_idx, None, doc_activity_tokens,
                                                    skip_first=skip_first, one_already_found=one_already_found)

    def _get_pet_entity_relation_rep(self, entity: Tuple[int, int, List[str], Optional[List[str]]], entity_type: str,
                                     source: bool = True):
        """
        return the dict representation of an entity for usage as part of a relation
        dict structure depends on the output format of the baseline (as in PET or simpler for benchmark library)
        :param entity: process entity as tuple -> sentence idx, token idx, ['Word', 'List'], optional(['word', 'List'])
                       last element of tuple is optional (is passed for gateways, for activities not)
        :param entity_type: entity type according to PET labels
        :param source: flag if it is source or target entity in the relation
        :return: Dictionary in format based on the output format
        """
        return self._get_pet_relation_rep(entity[0], entity[1], entity_type, entity[2], source=source)

    def _get_pet_relation_rep(self, sentence_idx: int, token_idx: int, entity_type: str, entity: List[str],
                              source: bool = True) -> Dict:
        """
        return the dict representation of an entity for usage as part of a relation
        :param sentence_idx: sentence index
        :param token_idx: token/word index
        :param entity_type: entity type according to PET labels
        :param entity: entity as list of single words
        :param source: flag if it is source or target entity in the relation
        :return: Dictionary in format based on the output format
        """
        if source:
            return {
                SOURCE_SENTENCE_ID: sentence_idx,
                SOURCE_HEAD_TOKEN_ID: token_idx,
                SOURCE_ENTITY_TYPE: entity_type,
                SOURCE_ENTITY: entity
            }
        else:
            return {
                TARGET_SENTENCE_ID: sentence_idx,
                TARGET_HEAD_TOKEN_ID: token_idx,
                TARGET_ENTITY_TYPE: entity_type,
                TARGET_ENTITY: entity
            }

    @staticmethod
    def _merge_flow(source, target):
        """
        merge two entity dictionaries (created with _get_entity_dict) into one
        :param source: source entity
        :param target: target entity
        :return: merged entitiy
        """
        return {**source[SOURCE], **target[TARGET]}

    def _get_entity_dict(self, entity: Tuple[int, int, List[str], Optional[List[str]]], entity_type: str) -> Dict:
        """
        create entity dictionary including the entity itself and its source and target repr. dicts for flow connections
        :param entity: entity (activity or gateway) in form of tuple
        :param entity_type: entity type
        :return: dict with structure
            element: Tuple[int, int, List[str], Optional[List[str]]]
            source/target: dict -> structure based on flow relation dict from PET
        """
        return {
            'element': entity,  # tuple (sentence idx, token idx, [word, list])
            'source': self._get_pet_entity_relation_rep(entity, entity_type, source=True) if entity else None,
            'target': self._get_pet_entity_relation_rep(entity, entity_type, source=False) if entity else None
        }

    def _log_gateway_frame(self, start_sentence_idx: int, start_token_idx: int, start_entity: Dict,
                           end_sentence_idx: int, end_token_idx: int, end_entity: Dict) -> None:
        """
        log frame of a gateway sequence defined by start/end sentence index, token index and entity
        """
        self._processed_doc_gateway_frames.append({
            START_SENTENCE_IDX: start_sentence_idx,
            START_TOKEN_ID: start_token_idx,
            START_ENTITY: start_entity,
            END_SENTENCE_IDX: end_sentence_idx,
            END_TOKEN_ID: end_token_idx,
            END_ENTITY: end_entity,
        })

    def _read_and_set_keywords(self) -> None:
        """
        load and set key word lists based on passed variant
        :return:
        """
        logger.info(f"Load keywords '{self.keywords}' ...")
        if self.keywords == LITERATURE:
            # based on key words proposals of Ferreira et al. 2017
            with open('data/keywords/literature_xor.txt') as file:
                self._xor_keywords = file.read().splitlines()

            with open('data/keywords/literature_and.txt') as file:
                self._and_keywords = file.read().splitlines()

        elif self.keywords == GOLD:
            self._xor_keywords = pet_reader.xor_key_words_gold
            self._and_keywords = pet_reader.and_key_words_gold

        elif self.keywords == OWN:
            raise NotImplementedError("Own keywords not implemented yet")

        self._xor_keywords.sort()
        self._and_keywords.sort()
        logger.info(f"Loaded {len(self._xor_keywords)} XOR and {len(self._and_keywords)} AND keywords")
        logger.info(f"Used XOR keywords: {self._xor_keywords}")
        logger.info(f"Used AND keywords: {self._and_keywords}")

    def _read_contradictory_gateways(self):
        """
        read pairs of contradictory exclusive gateway key words from file
        sort to prefer longer matching phrases during search
        :return:
        """
        with open('data/keywords/contradictory_gateways_gold.txt') as file:
            self._contradictory_gateways = [[x.split(" ") for x in l.strip().split(";")] for l in file.readlines()]
            self._contradictory_gateways.sort(key=lambda pair: len(pair[0]) + len(pair[1]), reverse=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    keyword_approach = KeywordApproach(approach_name='key_words_literature', keywords=LITERATURE,
                                       same_xor_gateway_threshold=1, output_format=BENCHMARK)
    # keyword_approach.evaluate_documents()

    # 'doc-1.1' for and gateway
    # 'doc-3.2' for exclusive gateway with two branches and overlapping concurrent gateway -> presentation candidate
    # 'doc-10.2' for or gateway in sentence
    # 'doc-9.5' for single exclusive gateway and two exclusive gateways with each two branches -> presentation candidate
    if True:
        doc_name = 'doc-3.2'

        xor_gateways, and_gateways, doc_flows, same_gateway_relations = keyword_approach.process_document(doc_name)

        print(" Concurrent gateways ".center(50, '-'))
        for gateway in and_gateways:
            print(gateway)

        print(" Exclusive gateways ".center(50, '-'))
        for gateway in xor_gateways:
            print(gateway)

        print(f" Flow relations ({len(doc_flows)}) of the whole document ".center(50, '-'))
        for flow_relation in doc_flows:
            print(flow_relation)

        print(" Same gateway relations ".center(50, '-'))
        for flow_relation in same_gateway_relations:
            print(flow_relation)
