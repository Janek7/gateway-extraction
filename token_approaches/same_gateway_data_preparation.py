import argparse
import itertools

import tensorflow as tf
from petreader.labels import *

from PetReader import pet_reader


def create_token_classification_dataset_cv(args: argparse.Namespace = None, shuffle: bool = True,
                                           context_sentences: int = 1, gateway_type: str = XOR_GATEWAY):
    # lists to store resultsCONCAT
    tokens = []  # tokenization results (either only text (if INDEX) or text concatenated with gateway n-grams (if CONCAT) )
    indexes = []  # index of gateway tokens in samples -> tuple
    labels = []  # labels (0 or 1)

    # A) GENERATE DATA
    for i, doc_name in enumerate(pet_reader.document_names):

        if doc_name != 'doc-2.1':
            continue

        print(doc_name)

        # 1) Prepare token data
        text = pet_reader.get_doc_text(doc_name)
        sample_ids = pet_reader.get_doc_sample_ids(doc_name)
        tokens = [list(zip(
            [s_i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
            [i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
            pet_reader.token_dataset.GetTokens(sample_id),
            pet_reader.token_dataset.GetNerTagLabels(sample_id))
        ) for s_i, sample_id in enumerate(sample_ids)]
        tokens_flattened = list(itertools.chain(*tokens))
        tokens_flattened = [(i,) + token_tuple for i, token_tuple in enumerate(tokens_flattened)]

        # 2) Identify gateway pairs
        # filter for B- tokens, because I-s do not mark a new gateway of interest
        gateway_tokens = [token_tuple for token_tuple in tokens_flattened if token_tuple[4] == f"B-{gateway_type}"]
        print()
        print('\n'.join(str(t) for t in gateway_tokens))
        print()
        gateway_pairs = [(gateway_tokens[i], gateway_tokens[i + 1]) for i in range(len(gateway_tokens) - 1)]
        # check if gateways are related
        same_gateway_relations = pet_reader.get_doc_relations(doc_name)[SAME_GATEWAY]

        for same_gateway_relation in same_gateway_relations:
            for key, value in same_gateway_relation.items():
                print(f"{key}: {value}")
            print()

        # list of labels if gateway are related (1) or not (0)
        pair_labels = []
        # check if for pair of two subsequent gateways exists a same gateway relation
        for g1, g2 in gateway_pairs:
            same_gateway_found = False
            for same_gateway_relation in same_gateway_relations:
                if not same_gateway_found \
                        and g1[1] == same_gateway_relation[SOURCE_SENTENCE_ID] \
                    and g1[2] == same_gateway_relation[SOURCE_HEAD_TOKEN_ID] \
                        and g2[1] == same_gateway_relation[TARGET_SENTENCE_ID] \
                        and g2[2] == same_gateway_relation[TARGET_HEAD_TOKEN_ID]:
                    pair_labels.append(1)
                    same_gateway_found = True
            if not same_gateway_found:
                pair_labels.append(0)

        for i in range(len(gateway_pairs)):
            g1, g2 = gateway_pairs[i]
            print(g1)
            print(g2)
            print(pair_labels[i])
            print()

        break


def create_full_training_dataset(args: argparse.Namespace, shuffle: bool = True) -> tf.data.Dataset:
    pass


if __name__ == '__main__':
    create_token_classification_dataset_cv()
