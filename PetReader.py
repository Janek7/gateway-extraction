import itertools
import collections
import os
from typing import List, Tuple, Dict
import logging

from petreader.RelationsExtraction import RelationsExtraction
from petreader.TokenClassification import TokenClassification
from petreader.labels import *

from utils import ROOT_DIR, load_pickle, save_as_pickle, flatten_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PetReader')


class PetReader:
    """
    Wrapper for PET dataset with own reading methods
    """

    def __init__(self):
        logger.info("Load RelationsExtraction dataset ...")
        self.relations_dataset = RelationsExtraction()
        logger.info("Load TokenClassification dataset ...")
        self.token_dataset = TokenClassification()

        self._xor_key_words_gold = self.get_gateway_key_words(self.token_dataset.GetXORGateways())
        self._and_key_words_gold = self.get_gateway_key_words(self.token_dataset.GetANDGateways())

    @staticmethod
    def get_gateway_key_words(dataset_gateway_list: List) -> List[str]:
        flattened = list(itertools.chain(*dataset_gateway_list))
        phrases = [" ".join(g).lower() for g in flattened]  # join phrases together if multiple words
        unique = list(set(phrases))
        return unique

    @property
    def xor_key_words_gold(self) -> List[str]:
        return self._xor_key_words_gold

    @property
    def and_key_words_gold(self) -> List[str]:
        return self._and_key_words_gold

    @property
    def document_names(self) -> List[str]:
        return self.token_dataset.GetDocumentNames()

    def get_document_number(self, doc_name):
        return self.relations_dataset.GetDocumentNumber(doc_name)

    def get_doc_text(self, doc_name: str) -> str:
        """
        return text of a document
        :param doc_name: doc name
        :return: doc text
        """
        return self.token_dataset.GetDocumentText(doc_name)

    def get_doc_sample_ids(self, doc_name: str) -> List[int]:
        """
        return the sample/sentence IDs belonging to the given doc_name
        :param doc_name: doc name
        :return: list of sample IDs
        """
        return self.token_dataset.get_n_sample_of_a_document(doc_name)

    def get_doc_sentences(self, doc_name: str) -> List[List[str]]:
        """
        return text of a document as list of samples/sentences
        :param doc_name: doc name
        :return: list of sentences
        """
        return [self.token_dataset.GetTokens(sample)
                for sample in self.token_dataset.get_n_sample_of_a_document(doc_name)]

    def get_doc_relations(self, doc_name: str) -> Dict:
        """
        return dictionary of relations of a document
        :param doc_name: doc name
        :return: relations dictionary
        """
        return self.relations_dataset.GetRelations(self.get_document_number(doc_name))

    def get_ner_tags(self, sample_number: int):
        return self.relations_dataset.GetSentencesWithIdsAndNerTagLabels(sample_number)

    def get_index_enriched_activities(self, doc_name: str) -> List[List[Tuple[str, int]]]:
        """
        Return activities of a document. Tokens get enriched with a position resulting in a tuple
        :param doc_name: document name
        :return: list of activities (represented as tuple) for each sentence
        """
        doc_activities = self.token_dataset.GetActivities(doc_name)
        doc_sentence_ner_labels = self.relations_dataset.GetSentencesWithIdsAndNerTagLabels(
            self.get_document_number(doc_name))

        doc_activity_tokens = []
        for i, (tokens, activities) in enumerate(zip(doc_sentence_ner_labels, doc_activities)):
            sentence_activity_tokens = []
            # log already assigned tokens to prevent double matching to one because of same activity text
            already_assigned_tokens = []
            # note: activity is a list because it could consist of more words (neglect here)
            for activity in activities:
                activity_token_triple = [token_triple for token_triple in tokens
                                         if token_triple[0] == activity[0]
                                         and token_triple not in already_assigned_tokens][0]
                already_assigned_tokens.append(activity_token_triple)
                sentence_activity_tokens.append((activity, activity_token_triple[1]))
            doc_activity_tokens.append(sentence_activity_tokens)
        return doc_activity_tokens

    def get_activities_in_relation_approach_format(self, doc_name: str) -> List[Tuple]:
        """
        return (flattened) list of activities of a document in format (sentence idx, word idx, token list)
        :param doc_name:
        :return:
        """
        activities = self.get_index_enriched_activities(doc_name)
        activities = [[(i, a[1], a[0], ACTIVITY) for a in sentence_activities] for i, sentence_activities in
                      enumerate(activities)]
        return flatten_list(activities)

    @property
    def most_common_activities(self) -> List[str]:
        """
        returns a list of all activities in data set ordered descended by their counts
        :return: list of activities
        """
        activities_flattened = [' '.join(a) for a in list(itertools.chain(*self.token_dataset.GetActivities()))]
        activity_counts = collections.Counter(activities_flattened)
        return [activity for activity, count in activity_counts.most_common()]

    def extract_gold_contradictory_keywords(self) -> List[Tuple[List[str], List[str]]]:
        """
        extract gold pairs of contradictory keywords from same gateway relations
        :return:
        """
        # 1) collect same gateway relations
        same_gateway_relations = []
        for doc_name in pet_reader.document_names:
            doc_relations = self.relations_dataset.GetRelations(self.get_document_number(doc_name))
            same_gateway_relations.extend(doc_relations[SAME_GATEWAY])
        # 2) reduce to unique lists of pairs
        contradictory_gateways = [([t.lower() for t in sg[SOURCE_ENTITY]],
                                   [t.lower() for t in sg[TARGET_ENTITY]]) for sg in same_gateway_relations]
        return contradictory_gateways


# Load / create and save pet_reader (for faster loading)

pet_reader_path = os.path.join(ROOT_DIR, "data/other/pet_reader.pkl")
if os.path.exists(pet_reader_path):
    logger.info(f"Reload pet_reader from {pet_reader_path}")
    pet_reader = load_pickle(pet_reader_path)
else:
    logger.info(f"Create pet_reader and save as {pet_reader_path}")
    pet_reader = PetReader()
    save_as_pickle(pet_reader, pet_reader_path)


if __name__ == '__main__':
    print(pet_reader.get_activities_in_relation_approach_format("doc-10.2"))
