import argparse
import logging
from typing import Tuple, List
import random

import tensorflow as tf
import transformers
from sklearn.model_selection import KFold
from transformers import BatchEncoding
from petreader.labels import *

from PetReader import pet_reader
from labels import *
from token_data_augmentation import get_synonym_samples
from utils import config, CURRENT_USED_SEED

logger = logging.getLogger('Data Preparation')


_tokenizer = transformers.AutoTokenizer.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
assert isinstance(_tokenizer, transformers.PreTrainedTokenizerFast)
all_sample_ids = pet_reader.token_dataset.GetRandomizedSampleNumbers()


# A) DATA SAMPLING

def get_samples(strategy: str = None, use_synonyms: bool = False) -> List[int]:
    """
    unified method to get list of samples to include in a dataset; which samples is controlled by strategy parameter
    use use_synonyms=True only with "normal" and "only gateway" strategy
    :param strategy: strategy which samples to include
    :param use_synonyms: flag if synonym samples should be included;
                         WARNING: True will change up/down sampling logic -> DO NOT USE TOGETHER
    :return: list of sample numbers
    """
    # extend all sample ids with created synonym sample ids
    if use_synonyms:
        synonym_samples = get_synonym_samples()
        all_sample_ids.extend(list(synonym_samples.keys()))

    # modify all_sample_ids list based on sampling strategy
    if strategy == NORMAL or strategy is None:
        return all_sample_ids
    elif strategy == UP_SAMPLING:
        return _up_sample_gateway_samples()
    elif strategy == DOWN_SAMPLING:
        return _down_sample_other_samples()
    elif strategy == ONLY_GATEWAYS:
        return _only_gateway_samples(use_synonyms=use_synonyms)
    else:
        raise ValueError(f"{strategy} is not a valid sampling strategy")


def _up_sample_gateway_samples() -> List[int]:
    """
    create a (shuffled) list of samples where gateway samples get upsampled to number of samples without gateway
    :return: list of sample ids
    """
    gateway_samples = _only_gateway_samples()
    without_gateway_samples = list(set(all_sample_ids) - set(gateway_samples))

    # sample samples with gateway until number of samples without gateway is reached
    upsampled_gateway_samples = []
    i = 0
    while len(upsampled_gateway_samples) < len(without_gateway_samples):
        upsampled_gateway_samples.append(gateway_samples[i])
        i += 1
        i %= len(gateway_samples)

    up_sampled_samples = without_gateway_samples + upsampled_gateway_samples
    random.seed(CURRENT_USED_SEED)
    random.shuffle(up_sampled_samples)
    return up_sampled_samples


def _down_sample_other_samples() -> List[int]:
    """
    create a (shuffled) list of samples where samples without gateway get down sampled to the number of samples with
    gateway
    :return: list of sample ids
    """
    gateway_samples = _only_gateway_samples()
    without_gateway_samples = list(set(all_sample_ids) - set(gateway_samples))
    # not all samples without gateway will be included -> shuffle to sample random ones
    random.seed(CURRENT_USED_SEED)
    random.shuffle(without_gateway_samples)

    # sample samples without gateway until number of samples with gateway is reached
    down_sampled_without_gateway_samples = []
    i = 0
    while len(down_sampled_without_gateway_samples) < len(gateway_samples):
        down_sampled_without_gateway_samples.append(without_gateway_samples[i])
        i += 1

    down_sampled_samples = gateway_samples + down_sampled_without_gateway_samples
    random.seed(CURRENT_USED_SEED)
    random.shuffle(down_sampled_samples)
    return down_sampled_samples


def _only_gateway_samples(use_synonyms: bool = False) -> List[int]:
    """
    return filtered list of samples ids that contain at least one gateway token
    :param use_synonyms: flag if synonym samples should be included
    """
    only_gateway_samples = [s for s in pet_reader.token_dataset.GetRandomizedSampleNumbers()
                            if f"B-{XOR_GATEWAY}" in pet_reader.token_dataset.GetSampleDictWithNerLabels(s)["ner-tags"]
                            or f"B-{AND_GATEWAY}" in pet_reader.token_dataset.GetSampleDictWithNerLabels(s)["ner-tags"]]
    if use_synonyms:
        synonym_samples = get_synonym_samples()
        only_gateway_samples.extend(list(synonym_samples.keys()))
    return only_gateway_samples


# B) DATASET CREATION


def preprocess_tokenization_data(sample_numbers: List = None, sampling_strategy: str = None, use_synonyms: bool = False,
                                 other_labels_weight: float = 0.1, label_set: str = 'filtered')\
        -> Tuple[BatchEncoding, tf.Tensor, tf.Tensor, List[List[int]]]:
    """
    create token classification samples from whole PET dataset -> samples (tokens) and their labels and weights for
    usage in a tensorflow dataset
    include either samples from sample_numbers list OR sample samples with sampling_strategy
    :param sample_numbers: list of concrete sample numbers
    :param sampling_strategy: strategy how to sample samples: 'normal', 'up', 'down', 'og' (only gateways)
    :param use_synonyms: flag if synonym samples should be included;
    :param other_labels_weight: sample weight to assign samples with tokens != gateway tokens
    :param label_set: flag if to use all labels ('all') or only gateway labels and one rest label ('filtered')
    :return: tokens, labels & weights as tensors, original word ids (2-dim integer list)
    """
    if sample_numbers and not sampling_strategy:
        sample_numbers = sample_numbers
    elif not sample_numbers and sampling_strategy:
        sample_numbers = get_samples(strategy=sampling_strategy, use_synonyms=use_synonyms)
    else:
        raise ValueError("Tokenization either based on conrete samples OR sampling strategy from whole data (not both)")

    # 1) prepare sample data
    sample_dicts = []
    synonym_samples = get_synonym_samples()
    for sample_number in sample_numbers:
        # in case sample is normal sample
        if sample_number < config[SYNONYM_SAMPLES_START_NUMBER]:
            sample_dicts.append(pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number))
        # in case sample is synonym sample
        else:
            sample_dicts.append(synonym_samples[sample_number])

    sample_sentences = [sample_dict['tokens'] for sample_dict in sample_dicts]

    # 2) transform tokens tags into IDs classification
    dataset_tokens = _tokenizer(sample_sentences, is_split_into_words=True, padding=True, return_tensors='tf')
    max_sentence_length = dataset_tokens['input_ids'].shape[1]

    # 3) transform NER token tags into labels for classification
    dataset_labels = []
    dataset_sample_weights = []
    dataset_word_ids = []
    for i, sample_dict in enumerate(sample_dicts):
        # tokenize again every single sample to get access to .word_ids()
        tokenization = _tokenizer(sample_dict['tokens'], is_split_into_words=True,
                                  padding='max_length', max_length=max_sentence_length, return_tensors='tf')
        sample_tokens = _tokenizer.convert_ids_to_tokens(tokenization['input_ids'][0])

        sample_labels = []
        sample_sample_weights = []
        # word index necessary, because one token in PET could be splitted into multiple tokens with tokenizer
        # multiple tokens have all the same word_id -> allows retrieval of the same one NER label from PET tokens
        for token, word_index in zip(sample_tokens, tokenization.word_ids()):
            # set special class for special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                sample_labels.append(TC_LABEL_OUT_OF_SCOPE)
                sample_sample_weights.append(TC_WEIGHTS_BERT_TOKENS)
            else:
                token_tag = sample_dict['ner-tags'][word_index]
                # XOR
                if token_tag.endswith(XOR_GATEWAY):
                    sample_labels.append(TC_LABEL_XOR)  # 2
                    sample_sample_weights.append(TC_WEIGHTS_GATEWAY_LABELS)
                # AND
                elif token_tag.endswith(AND_GATEWAY):
                    sample_labels.append(TC_LABEL_AND)  # 3
                    sample_sample_weights.append(TC_WEIGHTS_GATEWAY_LABELS)
                else:
                    if label_set == 'filtered':
                        sample_labels.append(TC_LABEL_OTHER)
                        sample_sample_weights.append(other_labels_weight)
                    else:
                        sample_sample_weights.append(other_labels_weight)
                        if token_tag.endswith("O"):
                            sample_labels.append(TC_LABEL_OTHER)
                        elif token_tag.endswith(ACTIVITY):
                            sample_labels.append(TC_LABEL_ACTIVITY)
                        elif token_tag.endswith(ACTIVITY_DATA):
                            sample_labels.append(TC_LABEL_ACTIVITY_DATA)
                        elif token_tag.endswith(ACTOR):
                            sample_labels.append(TC_LABEL_ACTOR)
                        elif token_tag.endswith(FURTHER_SPECIFICATION):
                            sample_labels.append(TC_LABEL_FURTHER_SPECIFICATION)
                        elif token_tag.endswith(CONDITION_SPECIFICATION):
                            sample_labels.append(TC_LABEL_CONDITION_SPECIFICATION)
                        else:
                            raise ValueError("Unexpected token tag:", token_tag)

        dataset_sample_weights.append(sample_sample_weights)
        dataset_labels.append(sample_labels)
        dataset_word_ids.append(tokenization.word_ids())

    dataset_labels = tf.constant(dataset_labels)
    dataset_sample_weights = tf.constant(dataset_sample_weights)
    return dataset_tokens, dataset_labels, dataset_sample_weights, dataset_word_ids


def _shuffle_tokenization_data(input_ids: tf.Tensor, attention_masks: tf.Tensor, labels: tf.Tensor,
                               sample_weights: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    shuffle tensors of tokenized data
    :return: data tensors in same format but shuffled
    """
    indices = tf.range(start=0, limit=input_ids.shape[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    input_ids = tf.gather(input_ids, shuffled_indices)
    attention_masks = tf.gather(attention_masks, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)
    sample_weights = tf.gather(sample_weights, shuffled_indices)
    return input_ids, attention_masks, labels, sample_weights


def _create_dataset(input_ids: tf.Tensor, attention_masks: tf.Tensor, labels: tf.Tensor, sample_weights: tf.Tensor)\
        -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks},
                                               labels,
                                               sample_weights))


def create_full_training_dataset(args: argparse.Namespace, shuffle: bool = True)\
        -> tf.data.Dataset:
    """
    create one training set of the whole data without separating a dev set
    :param args: args namespace
    :param shuffle: flag if shuffle the data
    :return: one tensorflow dataset
    """
    logger.info(f"Create full training dataset dataset (other_labels_weight: {args.other_labels_weight} - "
                f"label_set: {args.label_set} - batch_size: {args.batch_size} - shuffle: {shuffle})")
    tokens, labels, sample_weights, _ = preprocess_tokenization_data(use_synonyms=args.use_synonyms,
                                                                     sampling_strategy=args.sampling_strategy,
                                                                     other_labels_weight=args.other_labels_weight,
                                                                     label_set=args.label_set)
    dataset = _create_dataset(tokens["input_ids"], tokens["attention_mask"], labels, sample_weights)
    if args.batch_size:
        dataset = dataset.batch(args.batch_size)
    return dataset


def create_token_classification_dataset_cv(args: argparse.Namespace, shuffle: bool = True) \
        -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    create the dataset for token classification with huggingface transformers bert like models
    split into kfolds splits to use for cross validation
    :param args: args namespace
    :param shuffle: flag if shuffle the data
    :return: list of tuples (train, dev) as tf.data.Dataset objects
    """
    logger.info(f"Create CV (folds={args.kfolds}) dataset (other_labels_weight: {args.other_labels_weight} "
                f"- label_set: {args.label_set} - batch_size: {args.batch_size} - shuffle: {shuffle})")
    tokens, labels, sample_weights, _ = preprocess_tokenization_data(sampling_strategy=args.sampling_strategy,
                                                                     use_synonyms=args.use_synonyms,
                                                                     other_labels_weight=args.other_labels_weight,
                                                                     label_set=args.label_set)
    input_ids, attention_masks = tokens['input_ids'], tokens['attention_mask']

    # shuffle inputs before splitting in train/dev
    if shuffle:
        input_ids, attention_masks, labels, sample_weights = _shuffle_tokenization_data(input_ids, attention_masks,
                                                                                        labels, sample_weights)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=args.kfolds)

    # create folds
    folded_datasets = []
    for train, test in kfold.split(input_ids):
        train_tf_dataset = _create_dataset(tf.gather(input_ids, train),
                                           tf.gather(attention_masks, train),
                                           tf.gather(labels, train),
                                           tf.gather(sample_weights, train))
        dev_tf_dataset = _create_dataset(tf.gather(input_ids, test),
                                         tf.gather(attention_masks, test),
                                         tf.gather(labels, test),
                                         tf.gather(sample_weights, test))
        if args.batch_size:
            train_tf_dataset = train_tf_dataset.batch(args.batch_size)
            dev_tf_dataset = dev_tf_dataset.batch(args.batch_size)
        folded_datasets.append((train_tf_dataset, dev_tf_dataset))

    return folded_datasets


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if True:
        samples = get_samples(strategy="og", use_synonyms=True)
        print(len(samples))

