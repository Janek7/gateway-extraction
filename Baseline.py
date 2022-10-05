import logging
from typing import List, Tuple, Optional, Dict
from petreader.labels import *
from PetReader import pet_reader
from labels import *

logger = logging.getLogger('baseline')


class Baseline:

    def __init__(self, keywords: str = LITERATURE, output_format: str = BENCHMARK):
        """
        creates new instance of the baseline approach
        :param keywords: flag/variant which keywords to use, available: literature; gold; own
        :param output_format:
        """
        self.keywords = keywords
        self.output_format = output_format
        self._xor_keywords = None
        self._and_keywords = None
        self._set_keywords()

    def extract_gateways(self, sentence_list: List[str], gateway_type: str) \
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
        else:
            raise ValueError(f"Output format must be {PET} or {BENCHMARK}")

    def extract_exclusive_flow_relations(self, doc_activity_tokens: List[List[Tuple[str, int]]],
                                         extracted_gateways: List[List[Tuple[str, int, str]]]) -> List[List]:
        """
        ?
        :param doc_activity_tokens: list of activity tokens (word, idx) for each sentence
        :param extracted_gateways: list of own extracted gateway for each sentence

        :return: list of flow relations in source/target dict representation
        """
        raise NotImplementedError("XOR flow relation extraction not implemented yet")

    def extract_concurrent_flow_relations(self, doc_activity_tokens: List[List[Tuple[str, int]]],
                                          extracted_gateways: List[List[Tuple[str, int, str]]]) -> List[Dict]:
        """
        extract flow relations for already found AND gateways following the logic:
        - for every gateway, to extract parallel branches, add relation to next activity after and before, because
          that's the pattern how AND key phrases are usually used (oriented by rules of Ferreira et al. 2017)
        - for each case, check over borders if not found in same sentence
        - to extract the flow relation that points to the gateway merge point, take the second before
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

                # 1) Find related activities (previous and following are concurrent activities; second previous the one before the gateway)
                previous_activity = self._get_previous_activity(s_idx, gateway_lead_token[1], doc_activity_tokens)
                second_previous_activity = self._get_previous_activity(s_idx, gateway_lead_token[1],
                                                                       doc_activity_tokens, skip_first=True)
                following_activity = self._get_following_activity(s_idx, gateway_lead_token[1], doc_activity_tokens)

                # 2) Get representations for flow object dictionaries
                gateway_source_rep = self._get_relation_rep(s_idx, gateway_lead_token[1], AND_GATEWAY, gateway_entity,
                                                            source=True)
                gateway_target_rep = self._get_relation_rep(s_idx, gateway_lead_token[1], AND_GATEWAY, gateway_entity,
                                                            source=False)
                previous_activity_target_rep = self._get_relation_rep(previous_activity[0], previous_activity[1],
                                                                      ACTIVITY, previous_activity[2], source=False)
                second_previous_activity_target_rep = self._get_relation_rep(second_previous_activity[0],
                                                                             second_previous_activity[1], ACTIVITY,
                                                                             second_previous_activity[2], source=True)
                following_activity_target_rep = self._get_relation_rep(following_activity[0], following_activity[1],
                                                                       ACTIVITY, following_activity[2], source=False)

                # 3) Create relations (second previous -> gateway; gateway -> previous; gateway -> following)
                relations.append(self._merge_dicts(second_previous_activity_target_rep, gateway_target_rep))
                relations.append(self._merge_dicts(gateway_source_rep, previous_activity_target_rep))
                relations.append(self._merge_dicts(gateway_source_rep, following_activity_target_rep))

        return relations

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

        if previous_activities_sentence:
            # return when just searching the first last activity OR when one was already found before
            previous_activity = previous_activities_sentence[-1]
            # A) base case: activity found
            if not skip_first or one_already_found:
                return (sentence_idx, previous_activity[1], previous_activity[0])
            # B) recursive case: continue search for second previous activity at index of previous activity
            else:
                return self._get_previous_activity(sentence_idx, previous_activity[1], doc_activity_tokens,
                                                   one_already_found=True)
        # B) recursive case: continue search for previous activity in previous sentence
        else:
            next_sentence_idx = sentence_idx - 1
            # no sentences any more to search
            if next_sentence_idx == -1:
                return None
            # otherwise search recursively the previous sentence
            else:
                return self._get_previous_activity(next_sentence_idx, None, doc_activity_tokens,
                                                   skip_first=skip_first, one_already_found=one_already_found)

    def _get_following_activity(self, sentence_idx: int, token_idx: int,
                                doc_activity_tokens: List[List[Tuple[str, int]]]) -> Optional[Tuple[int, int, str]]:
        """
        search recursive for the next following activity from a start point defined by sentence_idx and token_idx
        :param sentence_idx: sentence index where to start the search
        :param token_idx: token index where to stat the search
        :param doc_activity_tokens: list of activity lists (describes whole document)

        :returns: triple of (sentence idx, token_idx, token) or None if none was found
        """
        # search for activities right to the token in target sentence if token is given else in the whole
        if token_idx is not None:
            following_activities_sentence = [a_t for a_t in doc_activity_tokens[sentence_idx] if a_t[1] > token_idx]
        else:
            following_activities_sentence = [a_t for a_t in doc_activity_tokens[sentence_idx]]

        # if activities were found, take the last one
        if following_activities_sentence:
            a_t = following_activities_sentence[-1]
            return (sentence_idx, a_t[1], a_t[0])
        else:
            next_sentence_idx = sentence_idx + 1
            # no sentences any more to search
            if next_sentence_idx == len(doc_activity_tokens):
                return None
            # otherwise search recursively the following sentence
            else:
                return self._get_following_activity(next_sentence_idx, None, doc_activity_tokens)

    def _get_relation_rep(self, sentence_idx: int, token_idx: int, entity_type: str, entity: List[str],
                          source: bool = True) -> Dict:
        """
        return the dict representation of an entity for usage as part of a flow relation
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

        else:
            raise ValueError(f"Output format of flow relation dictionaries must be {PET} or {BENCHMARK}")

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

        else:
            raise ValueError(f"Key words must be {LITERATURE} or {GOLD}")

        self._xor_keywords.sort()
        self._and_keywords.sort()
        logger.info(f"Loaded {len(self._xor_keywords)} XOR and {len(self._and_keywords)} AND keywords")
        logger.info(f"Used XOR keywords: {self._xor_keywords}")
        logger.info(f"Used AND keywords: {self._and_keywords}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    baseline = Baseline(keywords=LITERATURE, output_format=BENCHMARK)
    doc_id = 0

    and_gateways_full, and_gateways_benchmark = baseline.extract_gateways(pet_reader.get_doc_sentences(doc_id),
                                                                          AND_GATEWAY)
    for gateway in and_gateways_benchmark:
        print(gateway)

    and_flow_relations = baseline.extract_concurrent_flow_relations(
        doc_activity_tokens=pet_reader.get_index_enriched_activities(doc_id),
        extracted_gateways=and_gateways_full)

    for flow_relation in and_flow_relations:
        print(flow_relation)
