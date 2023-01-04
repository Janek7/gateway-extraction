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
import random
from typing import Tuple, List

import tensorflow as tf
import transformers
from sklearn.model_selection import KFold
from petreader.labels import *

from labels import *
from utils import config, ROOT_DIR, load_pickle, save_as_pickle
from PetReader import pet_reader
from token_approaches.token_data_augmentation import get_synonym_samples, get_synonyms_of_original_samples

logger = logging.getLogger('Data Preparation [Same Gateway CLS]')

_tokenizer = transformers.AutoTokenizer.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
assert isinstance(_tokenizer, transformers.PreTrainedTokenizerFast)


# load synonym data
synonym_samples = get_synonym_samples()
synonyms_of_original_samples = get_synonyms_of_original_samples()


def _get_doc_tokens_flattened(doc_name: str) -> Tuple[List[List], List[int]]:
    """
    extract, enrich and flatten tokens of given document
    :param doc_name: doc_name
    :return:
        - list of tuples -> (doc token index, sample id, sentence id, token id, token, ner-tag, #I-tokens)
        - list of sample_ids
    """
    sample_ids = pet_reader.get_doc_sample_ids(doc_name)
    doc_tokens = [list(zip(
        [sample_id for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
        [s_i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
        [i for i in range(len(pet_reader.token_dataset.GetTokens(sample_id)))],
        pet_reader.token_dataset.GetTokens(sample_id),
        pet_reader.token_dataset.GetNerTagLabels(sample_id))
    ) for s_i, sample_id in enumerate(sample_ids)]
    doc_tokens_flattened = list(itertools.chain(*doc_tokens))
    doc_tokens_flattened = [[i] + list(token_tuple) for i, token_tuple in enumerate(doc_tokens_flattened)]

    def get_following_i_tokens(token_index):
        """
        append number of following I- tokens in case of B- token for usage when computing n_grams
        :param token_index: token index
        :return: list of following I- tokens
        """
        following_i_tokens = []
        for token in doc_tokens_flattened[token_index + 1:]:
            if token[5].startswith("I-"):
                following_i_tokens.append(token)
            else:
                break
        return following_i_tokens

    doc_tokens_flattened = [doc_token + [len(get_following_i_tokens(doc_token[0]))]
                            for doc_token in doc_tokens_flattened]
    return doc_tokens_flattened, sample_ids


def _get_textual_token(token_tuple, gateways_sample_infos):
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


def _get_n_gram(token, n_gram, doc_tokens_flattened, gateways_sample_infos=None):
    """
    create n gram of a given token
    for gateway elements that consist of multiple tokens, include I- tokens as well by adding token[6] to range
    :param token: token tuple
    :param gateways_sample_infos: infos about which samples are used for surrounding gateways
    :return: textual n-gram
    """
    return ' '.join([_get_textual_token(token_tuple, gateways_sample_infos)
                     for token_tuple in doc_tokens_flattened[max(token[0] - n_gram, 0):
                                                             min(token[0] + n_gram + token[6] + 1,
                                                                 len(doc_tokens_flattened))]])


def _tokenize_textual_features(mode, texts, n_gram_tuples) -> transformers.BatchEncoding:
    """
    create a tokenization with different inputs based on passed mode
    :param mode: architecture variant / mode
    :param texts: texts
    :param n_gram_tuples: n gram tuples
    :return: encoded tokens
    """
    if mode == N_GRAM or mode == CONTEXT_LABELS_NGRAM:
        tokens = _tokenizer(n_gram_tuples, padding=True, return_tensors="tf")
    elif mode == CONTEXT_INDEX:
        tokens = _tokenizer(texts, padding=True, return_tensors='tf')
    elif mode == CONTEXT_NGRAM or mode == CONTEXT_TEXT_AND_LABELS_NGRAM:
        # tokenize text & pairs separately, because it is not possible to concat triple
        text_tokens = _tokenizer(texts, padding=True, return_tensors='tf')
        n_gram_tokens = _tokenizer(n_gram_tuples, padding=True, return_tensors="tf")
        # concat manually after (cut the CLS token of the second pair / n_grams)
        concatted_input_ids = tf.concat([text_tokens["input_ids"], n_gram_tokens["input_ids"][:, 1:]], axis=1)
        concatted_attention_masks = tf.concat([text_tokens["attention_mask"], n_gram_tokens["attention_mask"][:, 1:]],
                                              axis=1)
        tokens = transformers.BatchEncoding(
            {"input_ids": concatted_input_ids, "attention_mask": concatted_attention_masks})
    else:
        raise ValueError(f"mode must be {N_GRAM}, {CONTEXT_INDEX}, {CONTEXT_NGRAM}, {CONTEXT_LABELS_NGRAM} or"
                         f" {CONTEXT_TEXT_AND_LABELS_NGRAM}")

    return tokens


def _pad_context_labels(context_labels: List[int]) -> List[int]:
    """
    pad context labels to static maximum length from config (necessary for passing to dense prediction layer)
    :param context_labels: list of context labels unpadded
    :return: list of context labels padded
    """
    # pad context labels to same fixed length (pad with 0, label for activities = 1, label for other tokens = 2
    max_context = config[SAME_GATEWAY_CLASSIFIER][CONTEXT_LABEL_LENGTH]
    context_labels_padded = [row[:max_context] + [SGC_CONTEXT_LABEL_PADDING for i in range(max_context - len(row))]
                             for row in context_labels]
    return context_labels_padded


def _mask_activities(doc_tokens_flattened: List[List], masking_strategy: str) -> List[List]:
    """
    mask activities with "dummy", most common activity or most common activities (if multiple in one sentence)
    :param doc_tokens_flattened: list of tokens of a document
    :param masking_strategy: how activities should be masked
    :return: list of tokens with masked texts
    """
    found_activities = 0
    for token in doc_tokens_flattened:
        if token[5].endswith(ACTIVITY):
            if masking_strategy == DUMMY:
                masked = 'activity'
            elif masking_strategy == SINGLE_MASK:
                masked = pet_reader.most_common_activities[0]
            elif masking_strategy == MULTI_MASK:
                masked = pet_reader.most_common_activities[found_activities]
            found_activities += 1
            token[4] = masked
    return doc_tokens_flattened


def _get_gateway_pairs(gateway_type: str, doc_names: List[str] = []) -> List[Tuple]:
    pairs = []
    for i, doc_name in enumerate(pet_reader.document_names):

        if doc_names and (doc_name not in doc_names):
            continue

        if i % 5 == 0:
            print(f"processed {i} documents")

        # 1) Prepare token data
        doc_tokens_flattened, sample_ids = _get_doc_tokens_flattened(doc_name)

        # 2) Identify gateway pairs
        # filter for B- tokens, because I-s do not mark a new gateway of interest
        gateway_tokens = [token_tuple for token_tuple in doc_tokens_flattened if token_tuple[5] == f"B-{gateway_type}"]
        gateway_pairs = [(gateway_tokens[i], gateway_tokens[i + 1]) for i in range(len(gateway_tokens) - 1)]

        same_gateway_relations = pet_reader.get_doc_relations(doc_name)[SAME_GATEWAY]
        label = None  # if gateway are related (1) or not (0)
        # check if for pair of two subsequent gateways exists a same gateway relation
        for g1, g2 in gateway_pairs:
            same_gateway_found = False
            for same_gateway_relation in same_gateway_relations:
                if not same_gateway_found \
                        and g1[2] == same_gateway_relation[SOURCE_SENTENCE_ID] \
                        and g1[3] == same_gateway_relation[SOURCE_HEAD_TOKEN_ID] \
                        and g2[2] == same_gateway_relation[TARGET_SENTENCE_ID] \
                        and g2[3] == same_gateway_relation[TARGET_HEAD_TOKEN_ID]:
                    label = 1
                    same_gateway_found = True
            if not same_gateway_found:
                label = 0

        for pair in gateway_pairs:
            pairs.append((doc_name, pair, label))
    return pairs


def _create_dataset_sg(input_ids: tf.Tensor, attention_masks: tf.Tensor, indexes: tf.Tensor, context_labels: tf.Tensor,
                       labels: tf.Tensor) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(
        ({'input_ids': input_ids, 'attention_mask': attention_masks, "indexes": indexes,
          "context_labels": context_labels}, labels))


def _prepare_dataset_sg(cache_path: str, pairs: List[Tuple], mode: str, use_synonyms: bool = False,
                        activity_masking: str = NOT, context_sentences: int = 1, n_gram: int = 0) -> tf.data.Dataset:
    """
    extract and preprocess gateway pairs
    :param pairs: list of extracted gateway pairs (if not given, load in )
    :param context_sentences: context size = number of sentences before and after first and second gateway to include
    :param mode: flags for which SGC architecture the data is prepared
                    'n_gram_' -> concat of gateway token n_grams
                    'context_n_gram' -> concatenation of context and concat of gateway token n_grams
                    'context_index' -> context and indexes of gateway tokens in document
                    'context_labels_n_gram' -> concatenation of context token labels and concat of gateway token n_grams
    :param n_gram: n of n_grams to include from gateways in CONCAT mode
    :param use_synonyms: flag if synonym samples should be included;
    :return: tf dataset
    """
    # reload from cache if already exists
    if os.path.exists(cache_path):
        tokens, indexes, context_labels, labels = load_pickle(cache_path)
        logger.info("Reloaded same gateway data from cache")
        results = (tokens, indexes, context_labels, labels)

    else:

        # lists to store results
        texts = []  # context texts
        n_gram_tuples = []  # tuples of gateway n_grams (only necessary for mode=context_n_gram)
        indexes = []  # index of gateway tokens in samples -> tuple
        context_labels = []  # list of context token labels
        labels = []  # labels (0 or 1)

        for i, (doc_name, (g1, g2), label) in enumerate(pairs):
            doc_tokens_flattened, sample_ids = _get_doc_tokens_flattened(doc_name)

            if activity_masking in [DUMMY, SINGLE_MASK, MULTI_MASK]:
                doc_tokens_flattened = _mask_activities(doc_tokens_flattened, activity_masking)

            # Tokens/Text
            num_s = context_sentences
            sentences_in_scope = list(range(g1[2] - num_s if (g1[2] - num_s) > 0 else 0,
                                            g2[2] + num_s + 1 if (g2[2] + num_s + 1) < len(sample_ids) else len(
                                                sample_ids)))

            def append_not_token_data():
                """
                appending indexes, context_labels and labels of g1/g2 sample to dataset wide lists
                defined for reuse because of normal and synonym mode
                """
                # Indexes
                indexes.append((g1[0], g2[0]))
                # Context token labels
                context_labels.append([SGC_CONTEXT_LABEL_ACTIVITY if token[5] == ACTIVITY
                                       else SGC_CONTEXT_LABEL_OTHER for token in doc_tokens_flattened
                                       if token[2] in sentences_in_scope])
                # Label
                labels.append(label)

            if not use_synonyms:
                # Tokens/Text
                text_in_scope = ' '.join([token[4] for token in doc_tokens_flattened
                                          if token[2] in sentences_in_scope])
                texts.append(text_in_scope)
                if mode in [N_GRAM, CONTEXT_NGRAM, CONTEXT_LABELS_NGRAM, CONTEXT_TEXT_AND_LABELS_NGRAM]:
                    n_gram_tuples.append((_get_n_gram(g1, n_gram, doc_tokens_flattened),
                                          _get_n_gram(g2, n_gram, doc_tokens_flattened)))

                append_not_token_data()

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
                    text_in_scope = ' '.join([_get_textual_token(token, gateways_sample_infos)
                                              for token in doc_tokens_flattened if token[2] in sentences_in_scope])

                    texts.append(text_in_scope)
                    if mode in [N_GRAM, CONTEXT_NGRAM, CONTEXT_LABELS_NGRAM, CONTEXT_TEXT_AND_LABELS_NGRAM]:
                        n_gram_tuples.append(
                            (_get_n_gram(g1, n_gram, doc_tokens_flattened, gateways_sample_infos),
                             _get_n_gram(g2, n_gram, doc_tokens_flattened, gateways_sample_infos)))

                    append_not_token_data()

        results = (_tokenize_textual_features(mode, texts, n_gram_tuples),
                   tf.constant(indexes),
                   tf.constant(_pad_context_labels(context_labels)),
                   tf.constant(labels))

        # save in cache
        save_as_pickle(results, cache_path)

    return _create_dataset_sg(results[0]["input_ids"], results[0]["attention_mask"], results[1],
                              results[2], results[3])


def create_same_gateway_cls_dataset_full(gateway_type: str, args: argparse.Namespace) -> tf.data.Dataset:
    """
    create one training dataset of the whole data without separating a dev set
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param args: args namespace
    :return: one tensorflow dataset
    """
    logger.info(f"Create full same gateway classification dataset (batch_size={args.batch_size})")
    logger.info(f"Generate gateway data with data with params: mode={args.mode} - use_synonyms={args.use_synonyms}"
                f" - context={args.context_size} - n_gram={args.n_gram} - activity_masking={args.activity_masking}")

    # load samples to include in dataset
    pairs = _get_gateway_pairs(gateway_type)
    random.shuffle(pairs)
    logger.info(f"Basis are {len(pairs)} gateway pairs")

    # prepare path to store/reload from cache if already exists
    param_string = "reworked_" + '_'.join([str(p) for p in [gateway_type, args.use_synonyms, args.activity_masking,
                                                            args.mode, args.context_size, args.n_gram]])
    cache_path = os.path.join(ROOT_DIR, f"data/other/data_cache/same_gateway/same_gateway_data_{param_string}")

    # create dataset
    cache_path += "__fulldataset"
    tf_dataset = _prepare_dataset_sg(cache_path, pairs, args.mode, args.use_synonyms,
                                     args.activity_masking, args.context_size, args.n_gram)
    logger.info(f"Final Dataset -> {len(tf_dataset)}{f' ({len(pairs)} without syn.)' if args.use_synonyms else ''}")

    if args.batch_size:
        tf_dataset = tf_dataset.batch(args.batch_size)

    return tf_dataset


def create_same_gateway_cls_dataset_cv(gateway_type: str, args: argparse.Namespace = None) \
        -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    create the dataset for same gateway classification based on huggingface transformers bert like models
    split into kfolds splits to use for cross validation
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param args: args namespace
    :return: list of tuples (train, dev) as tf.data.Dataset objects
    """
    logger.info(
        f"Create same gateway classification cv dataset for {gateway_type}s (folds={args.folds} - batch_size={args.batch_size})")
    logger.info(f"Generate gateway data with data with params: mode={args.mode} - use_synonyms={args.use_synonyms}"
                f" - context={args.context_size} - n_gram={args.n_gram} - activity_masking={args.activity_masking}")

    # load samples to include in dataset
    pairs = _get_gateway_pairs(gateway_type)
    random.shuffle(pairs)
    logger.info(f"Basis are {len(pairs)} gateway pairs")

    # prepare path to store/reload from cache if already exists
    param_string = "reworked_" + '_'.join([str(p) for p in [gateway_type, args.use_synonyms, args.activity_masking,
                                                            args.mode, args.context_size, args.n_gram]])
    cache_path = os.path.join(ROOT_DIR, f"data/other/data_cache/same_gateway/same_gateway_data_{param_string}")

    # create datasets for k fold cross validation
    folded_datasets = []

    kfold = KFold(n_splits=args.folds)
    logger.info(f"Split data in {args.folds} folds")
    for i, (train, dev) in enumerate(kfold.split(pairs)):

        train_pairs = [p for i, p in enumerate(pairs) if i in train]
        dev_pairs = [p for i, p in enumerate(pairs) if i in dev]

        cache_path_train, cache_path_dev = f"{cache_path}__fold{i}_train", f"{cache_path}__fold{i}_dev"
        train_tf_dataset = _prepare_dataset_sg(cache_path_train, train_pairs, args.mode, args.use_synonyms,
                                               args.activity_masking, args.context_size, args.n_gram)
        dev_tf_dataset = _prepare_dataset_sg(cache_path_dev, dev_pairs, args.mode, False,
                                             args.activity_masking, args.context_size, args.n_gram)

        logger.info(
            f"Fold {i} -> {len(train_tf_dataset)}{f' ({len(train_pairs)} without syn.)' if args.use_synonyms else ''}"
            f"/ {len(dev_tf_dataset)}")

        if args.batch_size:
            train_tf_dataset = train_tf_dataset.batch(args.batch_size)
            dev_tf_dataset = dev_tf_dataset.batch(args.batch_size)

        folded_datasets.append((train_tf_dataset, dev_tf_dataset))

    return folded_datasets


def preprocess_gateway_pair(args: argparse.Namespace, doc_name: str, g1: Tuple, g2: Tuple) \
        -> Tuple[transformers.BatchEncoding, List[int], List[int]]:
    """
    create an input sample for inference based on two identified gateways
    :param args: args
    :param doc_name: doc_name
    :param g1: first gateway of pair to evaluate
    :param g2: second gateway of pair to evaluate
    :return: tokens, index, context_labels
    """
    texts = []
    indexes = []
    context_labels = []
    n_gram_tuples = []

    doc_tokens_flattened, sample_ids = _get_doc_tokens_flattened(doc_name)

    # find respective internal tokens (different representation) of passed g1 and g2
    g1_internal = None
    g2_internal = None
    i = 0
    while i < len(doc_tokens_flattened) and (g1_internal is None or g2_internal is None):
        internal_token = doc_tokens_flattened[i]
        # compare sentence idx, token idx and the token itself
        if internal_token[2] == g1[0] and internal_token[3] == g1[1] and internal_token[4] == g1[2][0]:
            g1_internal = internal_token
        if internal_token[2] == g2[0] and internal_token[3] == g2[1] and internal_token[4] == g2[2][0]:
            g2_internal = internal_token
        i += 1

    # Tokens/Text
    num_s = int(args.context_size)
    sentences_in_scope = list(range(g1_internal[2] - num_s if (g1_internal[2] - num_s) > 0 else 0,
                                    g2_internal[2] + num_s + 1 if (g2_internal[2] + num_s + 1) < len(sample_ids)
                                        else len(sample_ids)))
    text_in_scope = ' '.join([token[4] for token in doc_tokens_flattened
                              if token[2] in sentences_in_scope])
    texts.append(text_in_scope)

    # N-gram tuples
    if args.mode in [N_GRAM, CONTEXT_NGRAM, CONTEXT_LABELS_NGRAM, CONTEXT_TEXT_AND_LABELS_NGRAM]:
        n_gram_tuples.append((_get_n_gram(g1_internal, args.n_gram, doc_tokens_flattened),
                              _get_n_gram(g2_internal, args.n_gram, doc_tokens_flattened)))

    # Indexes
    indexes.append((g1_internal[0], g2_internal[0]))

    # Context token labels
    context_labels.append([SGC_CONTEXT_LABEL_ACTIVITY if token[5] == ACTIVITY
                           else SGC_CONTEXT_LABEL_OTHER for token in doc_tokens_flattened
                           if token[2] in sentences_in_scope])

    results = (_tokenize_textual_features(args.mode, texts, n_gram_tuples),
               tf.constant(indexes),
               tf.constant(_pad_context_labels(context_labels)))

    return results


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--folds", default=5, type=int, help="K folds.")
    parser.add_argument("--gateway", default=XOR_GATEWAY, type=str, help="Type of gateway to classify")
    parser.add_argument("--use_synonyms", default=True, type=str, help="Include synonym samples.")
    parser.add_argument("--activity_masking", default=MULTI_MASK, type=str, help="How to include activity data.")
    parser.add_argument("--context_size", default=1, type=int, help="Number of sentences around to include in text.")
    parser.add_argument("--mode", default=CONTEXT_TEXT_AND_LABELS_NGRAM, type=str,
                        help="How to include gateway information.")
    parser.add_argument("--n_gram", default=1, type=int, help="Number of tokens to include for gateway in CONCAT mode.")

    args_sg = parser.parse_args([] if "__file__" not in globals() else None)

    if True:
        folded_datasets_sg = create_same_gateway_cls_dataset_cv(XOR_GATEWAY, args_sg)
        for i, (train, dev) in enumerate(folded_datasets_sg):
            print(f"Fold {i}: train {len(train)} / dev {len(dev)}")

    if True:
        full_dataset_sg = create_same_gateway_cls_dataset_full(XOR_GATEWAY, args_sg)
        print(f"Full dataset size: {len(full_dataset_sg)}")

    # tokens, indexes, context_labels
    tokens, indexes, context_labels = preprocess_gateway_pair(args_sg, 'doc-1.1',
                                        g1=(2, 9, ['or'], ['or']),
                                        g2=(6, 0, ['If'], ['if']))
    print(tokens)
    print(indexes)
    print(context_labels)
