#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys
# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import itertools
import logging
import argparse
from typing import Tuple, List

import tensorflow as tf
import transformers
from sklearn.model_selection import KFold
from petreader.labels import *
from transformers import BatchEncoding

from labels import *
from utils import config, ROOT_DIR, load_pickle, save_as_pickle
from PetReader import pet_reader
from token_approaches.token_data_augmentation import get_synonym_samples, get_synonyms_of_original_samples

logger = logging.getLogger('Data Preparation [Same Gateway CLS]')

_tokenizer = transformers.AutoTokenizer.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
assert isinstance(_tokenizer, transformers.PreTrainedTokenizerFast)


def _create_dataset(input_ids: tf.Tensor, attention_masks: tf.Tensor, indexes: tf.Tensor, labels: tf.Tensor) \
        -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(
        ({'input_ids': input_ids, 'attention_mask': attention_masks, "indexes": indexes},
         labels))


def _shuffle_tokenization_data(input_ids: tf.Tensor, attention_masks: tf.Tensor, indexes: tf.Tensor, labels: tf.Tensor) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    shuffle tensors of tokenized data; seed for shuffling is seed_general from args
    :return: data tensors in same format but shuffled
    """
    indices = tf.range(start=0, limit=input_ids.shape[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    input_ids = tf.gather(input_ids, shuffled_indices)
    attention_masks = tf.gather(attention_masks, shuffled_indices)
    indexes = tf.gather(indexes, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)
    return input_ids, attention_masks, indexes, labels


def _preprocess_gateway_pairs(gateway_type: str, use_synonyms: bool = False, context_sentences: int = 1,
                              mode: str = CONCAT, n_gram: int = 1) -> Tuple[BatchEncoding, tf.Tensor, tf.Tensor]:
    """
    extract and preprocess gateway pairs
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param context_sentences: context size = number of sentences before and after first and second gateway to include
    :param mode: flag how to include gateway information (by concatenating n_grams of gateways to text or by indexes)
    :param n_gram: n of n_grams to include from gateways in CONCAT mode
    :param use_synonyms: flag if synonym samples should be included;
    :return: tokens as batch encoding, list of index pairs, list of labels
    """
    # reload from cache if already exists
    param_string = '_'.join([gateway_type, use_synonyms, mode, context_sentences, n_gram])
    cache_path = os.path.join(ROOT_DIR, f"data/other/same_gateway_data_{param_string}")
    if os.path.exists(cache_path):
        tokens, indexes, labels = load_pickle(cache_path)
        logger.info("Reloaded same gateway data from cache")
        return tokens, indexes, labels

    if use_synonyms:
        synonym_samples = get_synonym_samples()
        synonyms_of_original_samples = get_synonyms_of_original_samples()

    # lists to store results
    texts = []  # context texts
    n_gram_tuples = []  # tuples of gateway n_grams (only necessary for mode=CONCAT)
    indexes = []  # index of gateway tokens in samples -> tuple
    labels = []  # labels (0 or 1)

    # A) GENERATE DATA
    for i, doc_name in enumerate(pet_reader.document_names):

        if i % 5 == 0:
            print(f"processed {i} documents")

        # 1) Prepare token data
        text = pet_reader.get_doc_text(doc_name)
        sample_ids = pet_reader.get_doc_sample_ids(doc_name)
        doc_tokens = [list(zip(
            [sample_id for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
            [s_i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
            [i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
            pet_reader.token_dataset.GetTokens(sample_id),
            pet_reader.token_dataset.GetNerTagLabels(sample_id))
        ) for s_i, sample_id in enumerate(sample_ids)]
        doc_tokens_flattened = list(itertools.chain(*doc_tokens))
        doc_tokens_flattened = [(i,) + token_tuple for i, token_tuple in enumerate(doc_tokens_flattened)]
        # token represented as tuple: (doc token index, sample id, sentence id, token id, token, ner-tag)

        # 2) Identify gateway pairs
        # filter for B- tokens, because I-s do not mark a new gateway of interest
        gateway_tokens = [token_tuple for token_tuple in doc_tokens_flattened if token_tuple[5] == f"B-{gateway_type}"]
        gateway_pairs = [(gateway_tokens[i], gateway_tokens[i + 1]) for i in range(len(gateway_tokens) - 1)]

        # check if gateways are related
        same_gateway_relations = pet_reader.get_doc_relations(doc_name)[SAME_GATEWAY]
        pair_labels = []  # list of labels if gateway are related (1) or not (0)
        # check if for pair of two subsequent gateways exists a same gateway relation
        for g1, g2 in gateway_pairs:
            same_gateway_found = False
            for same_gateway_relation in same_gateway_relations:
                if not same_gateway_found \
                        and g1[2] == same_gateway_relation[SOURCE_SENTENCE_ID] \
                        and g1[3] == same_gateway_relation[SOURCE_HEAD_TOKEN_ID] \
                        and g2[2] == same_gateway_relation[TARGET_SENTENCE_ID] \
                        and g2[3] == same_gateway_relation[TARGET_HEAD_TOKEN_ID]:
                    pair_labels.append(1)
                    same_gateway_found = True
            if not same_gateway_found:
                pair_labels.append(0)

        # 3) prepare sample data
        def get_textual_token(token_tuple, gateways_sample_infos):
            """
            returns the textual token of the given token tuple considering the different possible samples (normal or synonyms)
            :param token_tuple: token tuple
            :param gateways_sample_infos: infos about which samples are used for surrounding gateways
            :returns: token
            """
            if not gateways_sample_infos:
                return token_tuple[4]

            (g1_sample_id, g1_sample_id_original), (g2_sample_id, g2_sample_id_original) = gateways_sample_infos

            # check if both gateways are in same sentence and token is in the sentence
            if g1_sample_id_original == g2_sample_id_original and token_tuple[1] == g1_sample_id_original:

                # prefer higher id to favor synonym samples (but all will be used once)
                sample_id_to_choose = max(g1_sample_id, g2_sample_id)
                if sample_id_to_choose >= config[SYNONYM_SAMPLES_START_NUMBER]:
                    return synonym_samples[sample_id_to_choose]['tokens'][token_tuple[3]]
                else:
                    return token_tuple[4]

            # if token is in sentence of first gateway
            elif token_tuple[1] == g1_sample_id_original:

                # if sample is original sample, take normal token
                if g1_sample_id == g1_sample_id_original:
                    return token_tuple[4]
                # if not, take token at the same index from synonym sample
                else:
                    return synonym_samples[g1_sample_id]['tokens'][token_tuple[3]]

            # if token is in sentence of second gateway
            elif token_tuple[1] == g2_sample_id_original:
                # if sample is original sample, take normal token
                if g2_sample_id == g2_sample_id_original:
                    return token_tuple[4]
                # if not, take token at the same index from synonym sample
                else:
                    return synonym_samples[g2_sample_id]['tokens'][token_tuple[3]]

            # if token is not in scope of gateway sentences but context -> return normal token
            else:
                return token_tuple[4]

        def get_n_gram(token, gateways_sample_infos=None):
            """
            create n gram of a given token
            :param token: token tuple
            :param gateways_sample_infos: infos about which samples are used for surrounding gateways
            :return: textual n-gram
            """
            return ' '.join([get_textual_token(token_tuple, gateways_sample_infos)
                             for token_tuple in doc_tokens_flattened[max(token[0] - n_gram, 0):
                                                                     min(token[0] + n_gram + 1,
                                                                         len(doc_tokens_flattened))]])

        for (g1, g2), label in zip(gateway_pairs, pair_labels):
            # Tokens/Text
            num_s = context_sentences
            sentences_in_scope = list(range(g1[2] - num_s if (g1[2] - num_s) > 0 else 0,
                                            g2[2] + num_s + 1 if (g2[2] + num_s + 1) < len(sample_ids) else len(
                                                sample_ids)))
            if not use_synonyms:
                # Tokens/Text
                text_in_scope = ' '.join([token[4] for token in doc_tokens_flattened
                                          if token[2] in sentences_in_scope])
                texts.append((text_in_scope))
                if mode == CONCAT:
                    n_gram_tuples.append((get_n_gram(g1), get_n_gram(g2)))

                # Indexes
                indexes.append((g1[0], g2[0]))

                # Label
                labels.append(label)

            else:
                # create cartesian product between different samples of sentences that include gateways
                # use for each gateway the sentence itself and optional synonyms
                if g1[1] == g2[1]:
                    gateway_sample_combinations = itertools.product(*[
                        [(g1[1], g1[1])],
                        [(g1[1], g1[1])] + [(s, g1[1]) for s in synonyms_of_original_samples[g1[1]]]])
                else:
                    g1_sample_ids = [(sample_id, g1[1]) for sample_id in [g1[1]] + synonyms_of_original_samples[g1[1]]]
                    g2_sample_ids = [(sample_id, g2[1]) for sample_id in [g2[1]] + synonyms_of_original_samples[g2[1]]]
                    gateway_sample_combinations = itertools.product(*[g1_sample_ids, g2_sample_ids])

                # iterate over pairs of gateway sentences (multiple possible if synonyms are used)
                for gateways_sample_infos in gateway_sample_combinations:
                    text_in_scope = ' '.join([get_textual_token(token, gateways_sample_infos)
                                              for token in doc_tokens_flattened if token[2] in sentences_in_scope])

                    texts.append(text_in_scope)
                    if mode == CONCAT:
                        n_gram_tuples.append(
                            (get_n_gram(g1, gateways_sample_infos), get_n_gram(g2, gateways_sample_infos)))

                    # Indexes
                    indexes.append((g1[0], g2[0]))

                    # Label
                    labels.append(label)

    # B) TOKENIZE TEXT
    if mode == INDEX:
        tokens = _tokenizer(texts, padding=True, return_tensors='tf')
    elif mode == CONCAT:
        # tokenize text & pairs seperately, because it is not possible to concat triple
        text_tokens = _tokenizer(texts, padding=True, return_tensors='tf')
        n_gram_tokens = _tokenizer(n_gram_tuples, padding=True, return_tensors="tf")
        # concat manually after (cut the CLS token of the second pair / n_grams)
        concatted_input_ids = tf.concat([text_tokens["input_ids"], n_gram_tokens["input_ids"][:, 1:]], axis=1)
        concatted_attention_masks = tf.concat([text_tokens["attention_mask"], n_gram_tokens["attention_mask"][:, 1:]],
                                              axis=1)
        tokens = transformers.BatchEncoding(
            {"input_ids": concatted_input_ids, "attention_mask": concatted_attention_masks})
    else:
        raise ValueError(f"mode must be {INDEX} or {CONCAT}")

    results = (tokens, tf.constant(indexes), tf.constant(labels))

    # save in cache
    save_as_pickle(results, cache_path)

    return results


def create_same_gateway_cls_dataset_full(gateway_type: str, args: argparse.Namespace = None, shuffle: bool = True) \
        -> tf.data.Dataset:
    """
    create one training set of the whole data without separating a dev set
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param args: args namespace
    :param shuffle: flag if shuffle the data
    :return: one tensorflow dataset
    """
    logger.info(f"Create full training dataset dataset (gateway type: {gateway_type} - batch_size: {args.batch_size} "
                f"- shuffle: {shuffle})")
    tokens, indexes, labels = _preprocess_gateway_pairs(gateway_type, use_synonyms=args.use_synonyms,
                                                        context_sentences=args.context_size, mode=args.mode,
                                                        n_gram=args.n_gram)
    input_ids, attention_masks = tokens['input_ids'], tokens['attention_mask']
    if shuffle:
        input_ids, attention_masks, indexes, labels = _shuffle_tokenization_data(input_ids, attention_masks, indexes,
                                                                                 labels)
    dataset = _create_dataset(input_ids, attention_masks, indexes, labels)

    if args.batch_size:
        dataset = dataset.batch(args.batch_size)
    return dataset


def create_same_gateway_cls_dataset_cv(gateway_type: str, args: argparse.Namespace = None, shuffle: bool = True) \
        -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    create the dataset for same gateway classification based on huggingface transformers bert like models
    split into kfolds splits to use for cross validation
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param args: args namespace
    :param shuffle: flag if shuffle the data
    :return: list of tuples (train, dev) as tf.data.Dataset objects
    """
    logger.info(f"Create CV (folds={args.folds}) dataset (gateway type: {gateway_type} - batch_size: {args.batch_size} "
                f"- shuffle: {shuffle})")
    tokens, indexes, labels = _preprocess_gateway_pairs(gateway_type, use_synonyms=args.use_synonyms,
                                                        context_sentences=args.context_size, mode=args.mode,
                                                        n_gram=args.n_gram)
    input_ids, attention_masks = tokens['input_ids'], tokens['attention_mask']
    if shuffle:
        input_ids, attention_masks, indexes, labels = _shuffle_tokenization_data(input_ids, attention_masks, indexes,
                                                                                 labels)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=5)

    # create folds
    folded_datasets = []
    for train, test in kfold.split(input_ids):
        train_tf_dataset = _create_dataset(tf.gather(input_ids, train),
                                           tf.gather(attention_masks, train),
                                           tf.gather(indexes, train),
                                           tf.gather(labels, train))
        dev_tf_dataset = _create_dataset(tf.gather(input_ids, test),
                                         tf.gather(attention_masks, test),
                                         tf.gather(indexes, test),
                                         tf.gather(labels, test))
        if args.batch_size:
            train_tf_dataset = train_tf_dataset.batch(args.batch_size)
            dev_tf_dataset = dev_tf_dataset.batch(args.batch_size)
        folded_datasets.append((train_tf_dataset, dev_tf_dataset))

    return folded_datasets


if __name__ == '__main__':

    # _preprocess_gateway_pairs(XOR_GATEWAY, context_sentences=1, mode=CONCAT, n_gram=1, use_synonyms=True)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default=XOR_GATEWAY, type=str, help="Type of gateway to classify")
    parser.add_argument("--use_synonyms", default=True, type=str, help="Include synonym samples.")
    parser.add_argument("--context_size", default=1, type=int, help="Number of sentences around to include in text.")
    parser.add_argument("--mode", default=CONCAT, type=str, help="How to include gateway information.")
    parser.add_argument("--n_gram", default=1, type=int, help="Number of tokens to include for gateway in CONCAT mode.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    dataset_full = create_same_gateway_cls_dataset_full(args.gateway, args, shuffle=True)

    from collections import Counter

    labels = [x[1].numpy() for x in dataset_full]
    print(Counter(labels))
