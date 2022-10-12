from petreader.labels import *
from PetReader import pet_reader
from petbenchmarks.benchmarks import BenchmarkApproach
from petbenchmarks.tokenclassification import TokenClassificationBenchmark
from petbenchmarks.relationsextraction import RelationsExtractionBenchmark
import logging
import json
import os
from typing import List, Tuple, Optional, Dict
from labels import *

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

        # check string parameters for valid values
        if self.keywords not in [LITERATURE, GOLD, OWN]:
            raise ValueError(f"Key words must be {LITERATURE} or {GOLD}")
        if self.output_format not in [PET, BENCHMARK]:
            raise ValueError(f"Output format must be {PET} or {BENCHMARK}")

    def evaluate_documents(self, doc_names: List[str] = None) -> None:
        """
        run extraction and evaluation with petbenchmarks
        :param doc_names: list of document names to evaluate, use all as default value
        :return: nothing, results are written to .json file
        """
        if not doc_names:
            doc_names = pet_reader.document_names
        logger.info(f"Start processing of {len(doc_names)} documents ...")

        # prepare evaluation structures to fill
        tcb = TokenClassificationBenchmark()
        process_elements = tcb.GetEmptyPredictionsDict()
        reb = RelationsExtractionBenchmark()
        relations = reb.GetEmptyPredictionsDict()

        # process all documents
        for i, doc_name in enumerate(doc_names):
            if i % 5 == 0:
                logger.info(f"Finished processing of {i} documents.")
            xor_gateways, and_gateways, doc_flows, same_gateway_relations = self.process_document(doc_name,
                                                                                                  output_format=BENCHMARK)
            process_elements[doc_name][XOR_GATEWAY].extend(xor_gateways)
            process_elements[doc_name][AND_GATEWAY].extend(and_gateways)
            relations[doc_name][FLOW].extend(doc_flows)
            relations[doc_name][SAME_GATEWAY].extend(same_gateway_relations)

        # save results as json
        folder = f'/data/results/{self.approach_name}/'
        logger.info(f"Save results to {folder}")
        process_elements_filename = os.path.join(folder, 'process_elements.json')
        relations_filename = os.path.join(folder, 'relations.json')
        with open(process_elements_filename, 'w') as file:
            json.dump(process_elements, file)
        with open(relations_filename, 'w') as file:
            json.dump(relations, file)

        # run evaluation
        logger.info(f"Run evaluation")
        BenchmarkApproach(approach_name=self.approach_name, predictions_file_or_folder=process_elements_filename)
        BenchmarkApproach(approach_name=self.approach_name, predictions_file_or_folder=relations_filename)

    def process_document(self, doc_name: str, output_format: str = None) -> Tuple[List, List, List, List]:
        """
        extracts and returns gateways and related flow relations for given document
        :param doc_name: document name
        :param output_format: optional output_format - by default self.output_format is used
                              parameter is introduced for necessary control in eval_all_documents
        :return: xor_gateways, and_gateways, doc_flows, same_gateway_relations
        """
        # temporary overwrite output_format in object
        output_format_saved = self.output_format
        if output_format:
            self.output_format = output_format

        # prepare document
        doc_sentences = pet_reader.get_doc_sentences(doc_name)
        doc_activities_enriched = pet_reader.get_index_enriched_activities(doc_name)

        # extract concurrent gateways and related flow relations
        and_gateways_pet, and_gateways_benchmark = self._extract_gateways(doc_sentences, AND_GATEWAY)
        and_flows = self._extract_concurrent_flows(doc_activities_enriched, and_gateways_pet)

        # extract exclusive gateways and related flow relations
        xor_gateways_pet, xor_gateways_benchmark = self._extract_gateways(doc_sentences, XOR_GATEWAY)
        xor_flows, same_gateway_relations = self._extract_exclusive_flows(doc_activities_enriched, xor_gateways_pet)

        # extract flow relations of gold activities and remove the ones involved in gateway flows
        gold_activity_flows = self._extract_gold_activity_flows(doc_activities_enriched)
        doc_flows = self._merge_flows(gold_activity_flows, xor_flows, and_flows)

        if self.output_format == PET:
            xor_gateways = xor_gateways_pet
            and_gateways = and_gateways_pet
        elif self.output_format == BENCHMARK:
            xor_gateways = xor_gateways_benchmark
            and_gateways = and_gateways_benchmark
        # overwrite again with saved value (constructor value)
        self.output_format = output_format_saved

        return xor_gateways, and_gateways, doc_flows, same_gateway_relations

    def _extract_gateways(self, sentence_list: List[str], gateway_type: str) \
            -> Tuple[List[List[Tuple[str, int, str]]], Optional[List[List[str]]]]:
        """
        extracts gateways in a key-word-based manner given a document structured in a list of sentences
        if two phrases would match to a token (e.g. 'in the meantime' and 'meantime'), the longer phrase is extracted
        :param sentence_list: document represented as list of sentences
        :param gateway_type: gateway type to extract ('XOR Gateway' or 'AND Gateway')

        :return: return a tuple: (first output is necessary even if output format is BENCHMARK, because positional
                                  information are necessary for flow extraction algorithm)
            1) a two dimensional list -> list of tuples (word, position in sentence, tag) for each sentence this
            produces the same structure as sentences and their tokens and NER labels are annotated in PET dataset
            2) if output format is BENCHMARK, return a list of gateways (each a list again); if not None
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
        pet_gateways = []  # for PET representation
        benchmark_gateways = []  # for BENCHMARK representation
        for s_idx, sentence in enumerate(sentence_list):
            sentence_gateways = []  # for PET representation
            sentence_to_search = f" {sentence.lower()} "  # lowercase and wrap with spaces for search of key words
            tokens = sentence.split(" ")
            tokens_lower = sentence.lower().split(" ")
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
                            if self.output_format == BENCHMARK:
                                benchmark_gateways.append([tokens[t_idx + i] for i, x in enumerate(key_phrase_tokens)])

            sentence_gateways.sort(key=lambda gateway_triple: gateway_triple[1])
            pet_gateways.append(sentence_gateways)

        if self.output_format == PET:
            return pet_gateways, None
        elif self.output_format == BENCHMARK:
            return pet_gateways, benchmark_gateways

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
            if self.output_format == BENCHMARK:
                flow_relations.append({SOURCE_ENTITY: a1[0], TARGET_ENTITY: a2[0]})
            elif self.output_format == PET:
                a1 = self._get_pet_relation_rep(s_idx_1, a1[1], ACTIVITY, a1[0], source=True)
                a2 = self._get_pet_relation_rep(s_idx_2, a2[1], ACTIVITY, a2[0], source=False)
                flow_relations.append(self._merge_dicts(a1, a2))
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

        gateways = self._preprocess_extracted_gateways(extracted_gateways)
        gateways_involved = []  # list for gateways already involved into sequence flows

        # RULE 1): check for every pair of following gateways if it fits to a gateway constellation with
        # contradictory key words. Gateways must be in range of same_xor_gateway_threshold sentences, otherwise they
        # would be seen as separate ones
        for i in range(len(gateways) - 1):
            g1, g2 = gateways[i], gateways[i + 1]
            # if sentence distances is larger than threshold, reject possible pair
            if abs(g2[0] - g1[0]) > self._same_xor_gateway_threshold:
                continue
            # check for every pair of following gateways if it fits to a gateway pair of contradictory key words
            for pattern_gateway_1, pattern_gateway_2 in self._contradictory_gateways:
                if g1[3] == pattern_gateway_1 and g2[3] == pattern_gateway_2 and g1[1] == 0:
                    gateways_involved.append(g1)
                    gateways_involved.append(g2)

                    # A) find related activities
                    pa_g1 = self._get_previous_activity(g1[0], g1[1], doc_activity_tokens)
                    fa_g1 = self._get_following_activity(g1[0], g1[1], doc_activity_tokens)
                    fa_g2 = self._get_following_activity(g2[0], g2[1], doc_activity_tokens)
                    # check if fol. activities of g1 and g2 are equal -> if yes, the first branch is without activity
                    if fa_g1 == fa_g2:
                        fa_g1 = 'empty branch'
                    ffa_g2 = self._get_following_activity(g2[0], g2[1], doc_activity_tokens, skip_first=True)

                    # B) get dictionary representations
                    g1_source = self._get_pet_entity_relation_rep(g1, XOR_GATEWAY, source=True)
                    g1_target = self._get_pet_entity_relation_rep(g1, XOR_GATEWAY, source=False)
                    g2_source = self._get_pet_entity_relation_rep(g2, XOR_GATEWAY, source=True)
                    g2_target = self._get_pet_entity_relation_rep(g2, XOR_GATEWAY, source=False)
                    if pa_g1:  # could be None if at document start
                        pa_g1_source = self._get_pet_entity_relation_rep(pa_g1, ACTIVITY, source=True)
                    if fa_g1 != 'empty branch' and fa_g1:  # could be set in 1) manually to empty branch or document end
                        fa_g1_source = self._get_pet_entity_relation_rep(fa_g1, ACTIVITY, source=True)
                        fa_g1_target = self._get_pet_entity_relation_rep(fa_g1, ACTIVITY, source=False)
                    if fa_g2:  # could be None if at document end
                        fa_g2_source = self._get_pet_entity_relation_rep(fa_g2, ACTIVITY, source=True)
                        fa_g2_target = self._get_pet_entity_relation_rep(fa_g2, ACTIVITY, source=False)
                    if ffa_g2:  # could be None if at document end
                        ffa_g2_target = self._get_pet_entity_relation_rep(ffa_g2, ACTIVITY, source=False)

                    # C.1) connect elements to sequence flows
                    # 1) previous activity to first gateway -> split point (if not None because of document start)
                    if pa_g1:
                        sequence_flows.append(self._merge_dicts(pa_g1_source, g1_target))
                    # 2) gateway 1 to following activity and following activity to activity after gateway (second
                    # following of g2)
                    # if None because of empty branch then directly there
                    if fa_g1:  # could be None if at document end
                        sequence_flows.append(self._merge_dicts(g1_source, fa_g1_target))
                        if ffa_g2:  # could be None if at document end
                            sequence_flows.append(self._merge_dicts(fa_g1_source, ffa_g2_target))
                    elif fa_g1 != 'empty branch' and ffa_g2:  # could be None if at document end
                        sequence_flows.append(self._merge_dicts(g1_source, ffa_g2_target))
                    # 3) gateway 2 to following activity and following activity to activity after gateway (second
                    # following of g2)
                    if fa_g2:  # could be None if at document end
                        sequence_flows.append(self._merge_dicts(g2_source, fa_g2_target))
                    if ffa_g2:  # could be None if at document end
                        sequence_flows.append(self._merge_dicts(fa_g2_source, ffa_g2_target))

                    # C.2) same gateway flows
                    same_gateway_relations.append(self._merge_dicts(g1_source, g2_target))

        # RULE 2): exclusive actions of common pattern "... <activity> ... or ... <activity> ..."
        for g in gateways:
            if g not in gateways_involved and g[3] == ['or']:
                # A) find related activities
                pa = self._get_previous_activity(g[0], g[1], doc_activity_tokens)
                ppa = self._get_previous_activity(g[0], g[1], doc_activity_tokens, skip_first=True)
                fa = self._get_following_activity(g[0], g[1], doc_activity_tokens)
                ffa = self._get_following_activity(g[0], g[1], doc_activity_tokens, skip_first=True)

                if pa and fa:  # check if existence because of document end/start
                    if pa[0] == g[0] and fa[0] == g[0]:  # check if in same sentence
                        # B) get dict representations
                        g_source = self._get_pet_entity_relation_rep(g, XOR_GATEWAY, source=True)
                        g_target = self._get_pet_entity_relation_rep(g, XOR_GATEWAY, source=False)
                        if pa:  # could be None if at document start
                            pa_source = self._get_pet_entity_relation_rep(pa, ACTIVITY, source=True)
                            pa_target = self._get_pet_entity_relation_rep(pa, ACTIVITY, source=False)
                        if fa:  # could be None if at document end
                            fa_source = self._get_pet_entity_relation_rep(fa, ACTIVITY, source=True)
                            fa_target = self._get_pet_entity_relation_rep(fa, ACTIVITY, source=False)
                        if ppa:  # could be None if at document start
                            ppa_source = self._get_pet_entity_relation_rep(ppa, ACTIVITY, source=True)
                        if ffa:  # could be None if at document end
                            ffa_target = self._get_pet_entity_relation_rep(ffa, ACTIVITY, source=False)

                        if pa is None or fa is None:
                            # if not surrounding activities are given, do not wire anything; TODO: maybe drop gateway
                            continue

                        # C) connect elements to sequence flows
                        # 1) second previous activity to gateway -> split point
                        if ppa:  # (if not None because of document start)
                            sequence_flows.append(self._merge_dicts(ppa_source, g_target))
                        # 2) gateway to following activity and previous activity -> exclusive branches
                        sequence_flows.append(self._merge_dicts(g_source, pa_target))
                        sequence_flows.append(self._merge_dicts(g_source, fa_target))
                        # 3) exclusive activities to second following activity of gateway -> merge point
                        if ffa:  # (if not None because of document end)
                            sequence_flows.append(self._merge_dicts(pa_source, ffa_target))
                            sequence_flows.append(self._merge_dicts(fa_source, ffa_target))

                        gateways_involved.append(g)

        # RULE 3): single-branch gateways: the gateway is related to an activity in the same sentence (order is arbitrary)
        # Assumptiosn: multi-branch gateways are already recognized by rule 1 before; only one activity for the gateway
        for g in gateways:
            if g not in gateways_involved and g[3] != ['or']:
                # A) find related activities
                pa = self._get_previous_activity(g[0], g[1], doc_activity_tokens)
                ppa = self._get_previous_activity(g[0], g[1], doc_activity_tokens, skip_first=True)
                fa = self._get_following_activity(g[0], g[1], doc_activity_tokens)
                ffa = self._get_following_activity(g[0], g[1], doc_activity_tokens, skip_first=True)

                # B) check if activity is before or after the gateway (assumption: both is not included)
                if fa[0] == g[0]:
                    case = 'activity after gateway'
                elif pa[0] == g[0]:
                    case = 'activity before gateway'
                else:
                    continue  # if no activity in same sentence, do not wire anything; TODO: maybe drop gateway again
                gateways_involved.append(g)

                # C) get dict representations
                g_source = self._get_pet_entity_relation_rep(g, XOR_GATEWAY, source=True)
                g_target = self._get_pet_entity_relation_rep(g, XOR_GATEWAY, source=False)
                if pa:  # could be None if at document start
                    pa_source = self._get_pet_entity_relation_rep(pa, ACTIVITY, source=True)
                    pa_target = self._get_pet_entity_relation_rep(pa, ACTIVITY, source=False)
                if fa:  # could be None if at document end
                    fa_source = self._get_pet_entity_relation_rep(fa, ACTIVITY, source=True)
                    fa_target = self._get_pet_entity_relation_rep(fa, ACTIVITY, source=False)
                if ppa:  # could be None if at document start
                    ppa_source = self._get_pet_entity_relation_rep(ppa, ACTIVITY, source=True)
                if ffa:  # could be None if at document end
                    ffa_target = self._get_pet_entity_relation_rep(ffa, ACTIVITY, source=False)

                # D) connect elements to sequence flows
                if case == 'activity after gateway':
                    # 1) previous activity to gateway -> split point
                    if pa:  # could be None if at document start
                        sequence_flows.append(self._merge_dicts(pa_source, g_target))
                    # 2) gateway to following activity -> exclusive branch
                    sequence_flows.append(self._merge_dicts(g_source, fa_target))
                    # 3) exclusive activity and gateway to second following activity of gateway -> merge point
                    if ffa:  # could be None if at document end
                        sequence_flows.append(self._merge_dicts(g_source, ffa_target))
                        sequence_flows.append(self._merge_dicts(fa_target, ffa_target))

                elif case == 'activity before gateway':
                    # 1) second previous activity to gateway -> split point
                    if ppa:  # could be None if at document start
                        sequence_flows.append(self._merge_dicts(ppa_source, g_target))
                    # 2) gateway to previous activity -> exclusive branch
                    sequence_flows.append(self._merge_dicts(g_source, pa_target))
                    # 3) exclusive activity and gateway to following activity of gateway -> merge point
                    if fa:  # could be None if at document end
                        sequence_flows.append(self._merge_dicts(g_source, fa_target))
                        sequence_flows.append(self._merge_dicts(pa_source, fa_target))
        if self.output_format == PET:
            sequence_flows.sort(key=lambda flow: flow['source-head-sentence-ID'])
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

        for g in self._preprocess_extracted_gateways(extracted_gateways):

            # 1) Find related activities (previous and following are concurrent activities; second previous the one
            # before the gateway)
            pa = self._get_previous_activity(g[0], g[1], doc_activity_tokens)
            ppa = self._get_previous_activity(g[0], g[1], doc_activity_tokens, skip_first=True)
            fa = self._get_following_activity(g[0], g[1], doc_activity_tokens)
            ffa = self._get_following_activity(g[0], g[1], doc_activity_tokens, skip_first=True)
            # 2) Get representations for flow object dictionaries
            g_source = self._get_pet_entity_relation_rep(g, AND_GATEWAY, source=True)
            g_target = self._get_pet_entity_relation_rep(g, AND_GATEWAY, source=False)
            if pa:  # could be None if at document start
                pa_target = self._get_pet_entity_relation_rep(pa, ACTIVITY, source=False)
                pa_source = self._get_pet_entity_relation_rep(pa, ACTIVITY, source=True)
            if ppa:  # could be None if at document start
                ppa_source = self._get_pet_entity_relation_rep(ppa, ACTIVITY, source=True)
            if fa:  # could be None if at document end
                fa_target = self._get_pet_entity_relation_rep(fa, ACTIVITY, source=False)
                fa_source = self._get_pet_entity_relation_rep(fa, ACTIVITY, source=True)
            if ffa:  # could be None if at document end
                ffa_target = self._get_pet_entity_relation_rep(ffa, ACTIVITY, source=False)

            # 3) Create relations
            # a) flow to gateway: second previous -> gateway
            if ppa:  # could be None if at document start
                relations.append(self._merge_dicts(ppa_source, g_target))
            # b) split into concurrent gateway branches: gateway -> previous; gateway -> following
            # following two None checks (probably) wont never be False, but for safety included
            if pa:  # could be None if at document start
                relations.append(self._merge_dicts(g_source, pa_target))
            if fa:  # could be None if at document end
                relations.append(self._merge_dicts(g_source, fa_target))
            # c) merge branches together: previous -> second following; following -> second following
            if ffa:  # could be None if at document end
                relations.append(self._merge_dicts(pa_source, ffa_target))
                relations.append(self._merge_dicts(fa_source, ffa_target))

        return relations

    @staticmethod
    def _preprocess_extracted_gateways(extracted_gateways):
        """
        flatten gateways but keep sentence index; merge multiple gateway tokens into one gateway
        :param extracted_gateways: gateways in PET format
        :return: flattened gateway list filled with (sentence_idx, start_token_idx, ['Word', 'List'], ['word', 'list'])
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
                    gateways.append((sentence_idx, start_token_idx, gateway_tokens, gateway_tokens_lower))
        return gateways

    @staticmethod
    def _merge_flows(gold_activity_flows: List[Dict], xor_flows: List[Dict], and_flows: List[Dict]) -> List[Dict]:
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
        print("xor", xor_flows)
        print("and", and_flows)
        gateway_flows_source_entities = [flow[SOURCE_ENTITY] for flow in gateway_flows]
        doc_flows = gateway_flows.copy()
        for flow in gold_activity_flows:
            if not flow[SOURCE_ENTITY] in gateway_flows_source_entities:
                doc_flows.append(flow)
        logger.info(f"{len(doc_flows)} doc flows")
        return doc_flows

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
        dict structure depends on the output format of the baseline (as in PET or simpler for benchmark library)
        :param sentence_idx: sentence index
        :param token_idx: token/word index
        :param entity_type: entity type according to PET labels
        :param entity: entity as list of single words
        :param source: flag if it is source or target entity in the relation
        :return: Dictionary in format based on the output format
        """
        if self.output_format == PET:
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

        elif self.output_format == BENCHMARK:
            if source:
                return {SOURCE_ENTITY: entity}
            else:
                return {TARGET_ENTITY: entity}

    @staticmethod
    def _merge_dicts(source_dict, target_dict):
        return {**source_dict, **target_dict}

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
        :return:
        """
        with open('data/keywords/contradictory_gateways_gold.txt') as file:
            self._contradictory_gateways = [[x.split(" ") for x in l.strip().split(";")] for l in file.readlines()]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    keyword_approach = KeywordApproach(approach_name='key_words_literature', keywords=LITERATURE,
                                       same_xor_gateway_threshold=1, output_format=PET)
    # keyword_approach.evaluate_documents()

    # 'doc-1.1' for and gateway
    # 'doc-3.2' for exclusive gateway with two branches and overlapping concurrent gateway!!
    # 'doc-10.2' for or gateway in sentence
    # 'doc-9.5' for single exclusive gateway and two exclusive gateways with each two branches -> Presentation candidate
    if True:
        doc_name = 'doc-9.5'

        xor_gateways, and_gateways, doc_flows, same_gateway_relations = keyword_approach.process_document(doc_name)

        print(" Concurrent gateways ".center(50, '-'))
        for gateway in and_gateways:
            print(gateway)

        print(" Exclusive gateways ".center(50, '-'))
        for gateway in xor_gateways:
            print(gateway)

        print(" Flow relations of the whole document ".center(50, '-'))
        for flow_relation in doc_flows:
            print(flow_relation)

        print(" Same gateway relations ".center(50, '-'))
        for flow_relation in same_gateway_relations:
            print(flow_relation)
