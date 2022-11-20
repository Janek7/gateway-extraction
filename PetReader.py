import itertools
import collections
import os
from typing import List, Tuple, Dict
from petreader.RelationsExtraction import RelationsExtraction
from petreader.TokenClassification import TokenClassification
import logging

from utils import ROOT_DIR, load_pickle, save_as_pickle

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
            # note: activity is a list because it could consist of more words (neglect here)
            for activity in activities:
                activity_token_triple = [token_triple for token_triple in tokens if token_triple[0] == activity[0]][0]
                sentence_activity_tokens.append((activity, activity_token_triple[1]))
            doc_activity_tokens.append(sentence_activity_tokens)
        return doc_activity_tokens

    @property
    def most_common_activities(self) -> List[str]:
        """
        returns a list of all activities in data set ordered descended by their counts
        :return: list of activities
        """
        activities_flattened = [' '.join(a) for a in list(itertools.chain(*self.token_dataset.GetActivities()))]
        activity_counts = collections.Counter(activities_flattened)
        return [activity for activity, count in activity_counts.most_common()]


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
    print(pet_reader.xor_key_words_gold)
