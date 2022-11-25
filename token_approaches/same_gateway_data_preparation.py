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

logger = logging.getLogger('Data Preparation [Same Gateway CLS]')

_tokenizer = transformers.AutoTokenizer.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
assert isinstance(_tokenizer, transformers.PreTrainedTokenizerFast)


# A) SAMPLING DATA
# TODO:


# B) DATASET CREATION


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


def _preprocess_gateway_pairs(gateway_type: str, context_sentences: int = 1, mode: str = CONCAT, n_gram: int = 1) \
        -> Tuple[BatchEncoding, tf.Tensor, tf.Tensor]:
    """
    extract and preprocess gateway pairs
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param context_sentences: context size = number of sentences before and after first and second gateway to include
    :param mode: flag how to include gateway information (by concatenating n_grams of gateways to text or by indexes)
    :param n_gram: n of n_grams to include from gateways in CONCAT mode
    :return: tokens as batch encoding, list of index pairs, list of labels
    """
    # reload from cache if already exists
    cache_path = os.path.join(ROOT_DIR,
                              f"data/other/same_gateway_data_{gateway_type}_{context_sentences}_{mode}_{n_gram}")
    if os.path.exists(cache_path):
        tokens, indexes, labels = load_pickle(cache_path)
        logger.info("Reloaded same gateway data from cache")
        return tokens, indexes, labels

    # lists to store results
    texts = []  # context texts
    n_gram_tuples = []  # tuples of gateway n_grams (only necessary for mode=CONCAT)
    indexes = []  # index of gateway tokens in samples -> tuple
    labels = []  # labels (0 or 1)

    # A) GENERATE DATA
    for i, doc_name in enumerate(pet_reader.document_names):

        if i % 5 == 0:
            logger.info(f"processed {i} documents")

        # 1) Prepare token data
        text = pet_reader.get_doc_text(doc_name)
        sample_ids = pet_reader.get_doc_sample_ids(doc_name)
        doc_tokens = [list(zip(
            [s_i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
            [i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
            pet_reader.token_dataset.GetTokens(sample_id),
            pet_reader.token_dataset.GetNerTagLabels(sample_id))
        ) for s_i, sample_id in enumerate(sample_ids)]
        doc_tokens_flattened = list(itertools.chain(*doc_tokens))
        doc_tokens_flattened = [(i,) + token_tuple for i, token_tuple in enumerate(doc_tokens_flattened)]

        # 2) Identify gateway pairs
        # filter for B- tokens, because I-s do not mark a new gateway of interest
        gateway_tokens = [token_tuple for token_tuple in doc_tokens_flattened if token_tuple[4] == f"B-{gateway_type}"]
        gateway_pairs = [(gateway_tokens[i], gateway_tokens[i + 1]) for i in range(len(gateway_tokens) - 1)]

        # check if gateways are related
        same_gateway_relations = pet_reader.get_doc_relations(doc_name)[SAME_GATEWAY]
        pair_labels = []  # list of labels if gateway are related (1) or not (0)
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

        # 3) prepare sample data
        def get_n_gram(token):
            return ' '.join([token_tuple[3] for token_tuple in
                             doc_tokens_flattened[max(token[0] - n_gram, 0):
                                                  min(token[0] + n_gram + 1, len(doc_tokens_flattened))]])

        for (g1, g2), label in zip(gateway_pairs, pair_labels):
            # Tokens/Text
            num_s = context_sentences
            sentences_in_scope = list(range(g1[1] - num_s if (g1[1] - num_s) > 0 else 0,
                                            g2[1] + num_s + 1 if (g2[1] + num_s + 1) < len(sample_ids) else len(
                                                sample_ids)))
            text_in_scope = ' '.join([token_tuple[3] for token_tuple in doc_tokens_flattened
                                      if token_tuple[1] in sentences_in_scope])
            texts.append((text_in_scope))
            if mode == CONCAT:
                n_gram_tuples.append((get_n_gram(g1), get_n_gram(g2)))

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
    tokens, indexes, labels = _preprocess_gateway_pairs(gateway_type, context_sentences=args.context_size,
                                                        mode=args.mode, n_gram=args.n_gram)
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
    tokens, indexes, labels = _preprocess_gateway_pairs(gateway_type, context_sentences=args.context_size,
                                                        mode=args.mode, n_gram=args.n_gram)
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
    folded_datasets = create_same_gateway_cls_dataset_cv(XOR_GATEWAY, None)

    for train, dev in folded_datasets:
        print(len(train), len(dev))
