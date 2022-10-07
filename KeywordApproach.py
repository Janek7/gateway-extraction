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

logger = logging.getLogger('baseline')


class KeywordApproach:

    def __init__(self, approach_name: str = None, keywords: str = LITERATURE, output_format: str = BENCHMARK):
        """
        creates new instance of the basic keyword approach
        :param approach_name: description of approach to use in result folder name; if not set use key word variant
        :param keywords: flag/variant which keywords to use; available: literature, gold, own
        :param output_format: output format of extracted element and flows; available: benchmark, pet
        """
        self.approach_name = approach_name
        if not self.approach_name:
            self.approach_name = f"keywords_{keywords}"
        self.keywords = keywords
        self.output_format = output_format
        self._xor_keywords = None
        self._and_keywords = None
        self._set_keywords()

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

    contradictory_gateways = [(['if'], ['otherwise']), (['if'], ['else'])]

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

        # helper method only for this method
        def preprocess_gateways(extracted_gateways):
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

        gateways = preprocess_gateways(extracted_gateways)

        for i in range(len(gateways) - 1):
            g1, g2 = gateways[i], gateways[i + 1]
            # check for every pair of following gateways if it fits to a gateway pair of contradictory key words
            for pattern_gateway_1, pattern_gateway_2 in KeywordApproach.contradictory_gateways:
                if g1[3] == pattern_gateway_1 and g2[3] == pattern_gateway_2:

                    # 1) find related activities
                    pa_g1 = self._get_previous_activity(g1[0], g1[1], doc_activity_tokens)
                    fa_g1 = self._get_following_activity(g1[0], g1[1], doc_activity_tokens)
                    fa_g2 = self._get_following_activity(g2[0], g2[1], doc_activity_tokens)
                    # check if fol. activities of g1 and g2 are equal -> if yes, the first branch is without activity
                    if fa_g1 == fa_g2:
                        fa_g1 = None
                    ffa_g2 = self._get_following_activity(g2[0], g2[1], doc_activity_tokens, skip_first=True)

                    # 2) get dictionary representations
                    g1_source = self._get_pet_relation_rep(g1[0], g1[1], XOR_GATEWAY, g1[2], source=True)
                    g1_target = self._get_pet_relation_rep(g1[0], g1[1], XOR_GATEWAY, g1[2], source=False)
                    g2_source = self._get_pet_relation_rep(g2[0], g2[1], XOR_GATEWAY, g2[2], source=True)
                    g2_target = self._get_pet_relation_rep(g2[0], g2[1], XOR_GATEWAY, g2[2], source=False)
                    pa_g1_source = self._get_pet_relation_rep(pa_g1[0], pa_g1[1], ACTIVITY, pa_g1[2], source=True)
                    if fa_g1:
                        fa_g1_source = self._get_pet_relation_rep(fa_g1[0], fa_g1[1], ACTIVITY, fa_g1[2], source=True)
                        fa_g1_target = self._get_pet_relation_rep(fa_g1[0], fa_g1[1], ACTIVITY, fa_g1[2], source=False)
                    fa_g2_source = self._get_pet_relation_rep(fa_g2[0], fa_g2[1], ACTIVITY, fa_g2[2], source=True)
                    fa_g2_target = self._get_pet_relation_rep(fa_g2[0], fa_g2[1], ACTIVITY, fa_g2[2], source=False)
                    ffa_g2_target = self._get_pet_relation_rep(ffa_g2[0], ffa_g2[1], ACTIVITY, ffa_g2[2], source=False)

                    # 3.1) connect elements to sequence flows
                    # a) previous activity to first gateway -> split point
                    sequence_flows.append(self._merge_dicts(pa_g1_source, g1_target))
                    # b) gateway 1 to fol. activity and fol. activity to activity after gateway (second fol. of g2)
                    # if None directly there because of empty branch
                    if fa_g1:
                        sequence_flows.append(self._merge_dicts(g1_source, fa_g1_target))
                        sequence_flows.append(self._merge_dicts(fa_g1_source, ffa_g2_target))
                    else:
                        sequence_flows.append(self._merge_dicts(g1_source, ffa_g2_target))
                    # c) gateway 2 to fol. activity and fol. activity to activity after gateway (second fol. of g2)
                    sequence_flows.append(self._merge_dicts(g2_source, fa_g2_target))
                    sequence_flows.append(self._merge_dicts(fa_g2_source, ffa_g2_target))

                    # 3.2) same gateway flows
                    same_gateway_relations.append(self._merge_dicts(g1_source, g2_target))

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
        - assumption: only one parallel gateway per sentence

        :param doc_activity_tokens: list of activity tokens (word, idx) for each sentence
        :param extracted_gateways: list of own extracted gateway for each sentence
        :return: list of flow relations in source/target dict representation
        """
        relations = []

        for s_idx, gateways in enumerate(extracted_gateways):
            if gateways:
                # assume only one gateway
                gateway_lead_token = gateways[0]
                gateway_entity = [g[0] for g in gateways]

                # 1) Find related activities (previous and following are concurrent activities; second previous the one
                # before the gateway)
                previous_activity = self._get_previous_activity(s_idx, gateway_lead_token[1], doc_activity_tokens)
                second_previous_activity = self._get_previous_activity(s_idx, gateway_lead_token[1],
                                                                       doc_activity_tokens, skip_first=True)
                following_activity = self._get_following_activity(s_idx, gateway_lead_token[1], doc_activity_tokens)
                second_following_activity = self._get_following_activity(s_idx, gateway_lead_token[1],
                                                                         doc_activity_tokens, skip_first=True)
                # 2) Get representations for flow object dictionaries
                gateway_source_rep = self._get_pet_relation_rep(s_idx, gateway_lead_token[1], AND_GATEWAY,
                                                                gateway_entity,
                                                                source=True)
                gateway_target_rep = self._get_pet_relation_rep(s_idx, gateway_lead_token[1], AND_GATEWAY,
                                                                gateway_entity,
                                                                source=False)
                previous_activity_target_rep = self._get_pet_relation_rep(previous_activity[0], previous_activity[1],
                                                                          ACTIVITY, previous_activity[2], source=False)
                previous_activity_source_rep = self._get_pet_relation_rep(previous_activity[0], previous_activity[1],
                                                                          ACTIVITY, previous_activity[2], source=True)
                second_previous_activity_target_rep = self._get_pet_relation_rep(second_previous_activity[0],
                                                                                 second_previous_activity[1], ACTIVITY,
                                                                                 second_previous_activity[2],
                                                                                 source=True)
                following_activity_target_rep = self._get_pet_relation_rep(following_activity[0], following_activity[1],
                                                                           ACTIVITY, following_activity[2],
                                                                           source=False)
                following_activity_source_rep = self._get_pet_relation_rep(following_activity[0], following_activity[1],
                                                                           ACTIVITY, following_activity[2], source=True)
                second_following_activity_target_rep = self._get_pet_relation_rep(second_following_activity[0],
                                                                                  second_following_activity[1],
                                                                                  ACTIVITY,
                                                                                  second_following_activity[2],
                                                                                  source=False)

                # 3) Create relations
                # a) flow to gateway: second previous -> gateway
                relations.append(self._merge_dicts(second_previous_activity_target_rep, gateway_target_rep))
                # b) split into concurrent gateway branches: gateway -> previous; gateway -> following
                relations.append(self._merge_dicts(gateway_source_rep, previous_activity_target_rep))
                relations.append(self._merge_dicts(gateway_source_rep, following_activity_target_rep))
                # c) merge branches together: previous -> second following; following -> second following
                relations.append(self._merge_dicts(previous_activity_source_rep, second_following_activity_target_rep))
                relations.append(self._merge_dicts(following_activity_source_rep, second_following_activity_target_rep))

        return relations

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

    def _get_pet_relation_rep(self, sentence_idx: int, token_idx: int, entity_type: str, entity: List[str],
                              source: bool = True) -> Dict:
        """
        return the dict representation of an entity for usage as part of a relation
        dict structure depends on the output format of the baseline (as in PET or simpler for benchmark library)
        :param sentence_idx: sentence index
        :param token_idx: token/word index
        :param entity_type: entity type according to PET labels
        :param entity: entity as list of single words
        :param source: flag if the
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

    def _set_keywords(self) -> None:
        """
        load and set key word lists based on passed variant
        :return:
        """
        logger.info(f"Load keywords '{self.keywords}' ...")
        if self.keywords == LITERATURE:
            # based on key words proposals of Ferreira et al. 2017
            with open('data/keywords/literature_xor.txt') as f:
                self._xor_keywords = f.read().splitlines()

            with open('data/keywords/literature_and.txt') as f:
                self._and_keywords = f.read().splitlines()

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    keyword_approach = KeywordApproach(approach_name='baseline_literature', keywords=LITERATURE,
                                       output_format=BENCHMARK)

    # keyword_approach.evaluate_documents()

    # 'doc-1.1' for and gateway
    # 'doc-3.2' for exclusive gateway with two branches and if
    if True:
        doc_name = 'doc-3.2'

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
