import itertools
from typing import List, Tuple
from petreader.RelationsExtraction import RelationsExtraction
from petreader.TokenClassification import TokenClassification


class PetReader:
    _relations_dataset = RelationsExtraction()
    _token_dataset = TokenClassification()

    def __init__(self):
        self._xor_key_words_gold = self.get_gateway_key_words(PetReader._token_dataset.GetXORGateways())
        self._and_key_words_gold = self.get_gateway_key_words(PetReader._token_dataset.GetANDGateways())

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
        return self._token_dataset.GetDocumentNames()

    @staticmethod
    def get_document_number(doc_name):
        return PetReader._relations_dataset.GetDocumentNumber(doc_name)

    @staticmethod
    def get_doc_text(doc_name: str) -> str:
        """
        return text of a document
        :param doc_name: doc name
        :return: doc text
        """
        return PetReader._token_dataset.GetDocumentText(doc_name)

    @staticmethod
    def get_doc_sentences(doc_name: str) -> List[str]:
        """
        return text of a document as list of sentences
        :param doc_name: doc name
        :return: list of sentences
        """
        return [sentence.strip() for sentence in PetReader.get_doc_text(doc_name).split(".") if sentence.strip() != ""]

    def get_index_enriched_activities(self, doc_name: str) -> List[List[Tuple[str, int]]]:
        """
        Return activities of a document. Tokens get enriched with a position resulting in a tuple
        :param doc_id: document id as integer
        :return: list of activities (represented as tuple) for each sentence
        """
        doc_activities = PetReader._token_dataset.GetDocumentActivities(doc_name)
        doc_sentence_ner_labels = PetReader._relations_dataset.GetSentencesWithIdsAndNerTagLabels(
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


pet_reader = PetReader()
