import itertools
from typing import List, Tuple
from petreader.RelationsExtraction import RelationsExtraction
from petreader.TokenClassification import TokenClassification


class PetReader:
    relations_dataset = RelationsExtraction()
    token_dataset = TokenClassification()

    def __init__(self):
        self._xor_key_words_gold = self.get_gateway_key_words(PetReader.token_dataset.GetXORGateways())
        self._and_key_words_gold = self.get_gateway_key_words(PetReader.token_dataset.GetANDGateways())

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

    def get_doc_sentences(self, doc_name: str) -> List[List[str]]:
        """
        return text of a document as list of samples/sentences
        :param doc_name: doc name
        :return: list of sentences
        """
        return [self.token_dataset.GetTokens(sample)
                for sample in self.token_dataset.get_n_sample_of_a_document(doc_name)]

    def get_ner_tags(self, sample_number: int):
        return self.relations_dataset.GetSentencesWithIdsAndNerTagLabels(sample_number)

    def get_index_enriched_activities(self, doc_name: str) -> List[List[Tuple[str, int]]]:
        """
        Return activities of a document. Tokens get enriched with a position resulting in a tuple
        :param doc_id: document id as integer
        :return: list of activities (represented as tuple) for each sentence
        """
        doc_activities = PetReader.token_dataset.GetDocumentActivities(doc_name)
        doc_sentence_ner_labels = PetReader.relations_dataset.GetSentencesWithIdsAndNerTagLabels(
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
