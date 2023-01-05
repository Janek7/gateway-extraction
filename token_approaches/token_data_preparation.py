# add parent dir to sys path for import of modules
import os
import sys
# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import argparse
import logging
from typing import Tuple, List, Dict
import random

import tensorflow as tf
import transformers
from sklearn.model_selection import KFold
from transformers import BatchEncoding
from petreader.labels import *

from PetReader import pet_reader
from labels import *
from token_approaches.token_data_augmentation import get_synonym_samples, get_synonyms_of_original_samples
from utils import config, CURRENT_USED_SEED

logger = logging.getLogger('Data Preparation [Token CLS]')


_tokenizer = transformers.AutoTokenizer.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
assert isinstance(_tokenizer, transformers.PreTrainedTokenizerFast)

# load synonym data
synonym_samples = get_synonym_samples()
synonyms_of_original_samples = get_synonyms_of_original_samples()


# SAMPLING STRATEGIES -> provide list of sample IDs to use

def _get_sample_ids(strategy: str = None) -> List[int]:
    """
    unified method to get list of samples to include in a dataset; which samples is controlled by strategy parameter
    use use_synonyms=True only with "normal" and "only gateway" strategy
    :param strategy: strategy which samples to include
    :return: list of sample numbers
    """
    all_sample_ids = pet_reader.token_dataset.GetRandomizedSampleNumbers()

    # modify all_sample_ids list based on sampling strategy
    if strategy == NORMAL or strategy is None:
        return all_sample_ids
    elif strategy == UP_SAMPLING:
        return _up_sample_gateway_samples(all_sample_ids)
    elif strategy == DOWN_SAMPLING:
        return _down_sample_other_samples(all_sample_ids)
    elif strategy == ONLY_GATEWAYS:
        return _only_gateway_samples()
    else:
        raise ValueError(f"{strategy} is not a valid sampling strategy")


def _up_sample_gateway_samples(all_sample_ids: List[int]) -> List[int]:
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


def _down_sample_other_samples(all_sample_ids: List[int]) -> List[int]:
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


def _only_gateway_samples() -> List[int]:
    """
    return filtered list of samples ids that contain at least one gateway token
    """
    only_gateway_samples = [s for s in pet_reader.token_dataset.GetRandomizedSampleNumbers()
                            if f"B-{XOR_GATEWAY}" in pet_reader.token_dataset.GetSampleDictWithNerLabels(s)["ner-tags"]
                            or f"B-{AND_GATEWAY}" in pet_reader.token_dataset.GetSampleDictWithNerLabels(s)["ner-tags"]]
    return only_gateway_samples


# OTHER UTILITY METHODS


def _create_dataset(input_ids: tf.Tensor, attention_masks: tf.Tensor, labels: tf.Tensor, sample_weights: tf.Tensor) \
        -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks},
                                               labels,
                                               sample_weights))


def _mask_activities(sample_dicts: List[Dict], masking_strategy: str) -> List[Dict]:
    """
    mask activities with "dummy", most common activity or most common activities (if multiple in one sentence)
    :param sample_dicts: list of samples represented as dictionaries (including tokens and ner-tags)
    :param masking_strategy: how activities should be asked
    :return: list of sample dictionaries with masked tokens
    """
    for dictionary in sample_dicts:
        found_activities = 0
        masked_tokens = []
        for token, tag in zip(dictionary["tokens"], dictionary["ner-tags"]):
            if tag.endswith(ACTIVITY):
                if masking_strategy == DUMMY:
                    token = 'activity'
                elif masking_strategy == SINGLE_MASK:
                    token = pet_reader.most_common_activities[0]
                elif masking_strategy == MULTI_MASK:
                    token = pet_reader.most_common_activities[found_activities]
                found_activities += 1
            masked_tokens.append(token)
        dictionary["tokens"] = masked_tokens
    return sample_dicts


# DATA GENERATION


def prepare_token_cls_data(sample_numbers: List[int], other_labels_weight: float = 0.1,
                           label_set: str = 'filtered', activity_masking: str = None) \
        -> Tuple[BatchEncoding, tf.Tensor, tf.Tensor, List[List[int]]]:
    """
    create token classification samples from whole PET dataset -> samples (tokens) and their labels and weights for
    usage in a tensorflow dataset
    include either samples from sample_numbers list OR sample samples with sampling_strategy
    :param sample_numbers: list of concrete sample numbers
    :param other_labels_weight: sample weight to assign samples with tokens != gateway tokens
    :param label_set: flag if to use all labels ('all') or only gateway labels and one rest label ('filtered')
    :param activity_masking: flag how to use activity data in tokenization
    :return: tokens, labels & weights as tensors, original word ids (2-dim integer list)
    """

    # 1) prepare sample data
    sample_dicts = []
    for sample_number in sample_numbers:
        # in case sample is normal sample
        if sample_number < config[SYNONYM_SAMPLES_START_NUMBER]:
            sample_dicts.append(pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number))
        # in case sample is synonym sample
        else:
            sample_dicts.append(synonym_samples[sample_number])

    # apply optional activity masking
    if activity_masking in [SINGLE_MASK, MULTI_MASK]:
        sample_dicts = _mask_activities(sample_dicts, activity_masking)

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


def create_token_cls_dataset_full(args: argparse.Namespace) -> tf.data.Dataset:
    """
    create one training dataset of the whole data without separating a dev set
    :param args: args namespace
    :return: one tensorflow dataset
    """
    logger.info(f"Create full token classification dataset (batch_size={args.batch_size})")

    # load samples to include in dataset
    sample_ids = _get_sample_ids(strategy=args.sampling_strategy)
    random.shuffle(sample_ids)
    logger.info(
        f"Generate token data with params: sampling_strategy={args.sampling_strategy} - use_synonyms={args.use_synonyms}"
        f" - labels={args.labels} - other_labels_weight={args.other_labels_weight}")
    logger.info(f"Basis are {len(sample_ids)} samples from strategy '{args.sampling_strategy}'")

    # include synonyms in samples
    samples_number_old = len(sample_ids)
    if args.use_synonyms:
        synonym_samples = [synonyms for original_sample_id, synonyms in synonyms_of_original_samples.items()
                           if original_sample_id in sample_ids]
        synonym_samples_flattened = [item for sublist in synonym_samples for item in sublist]
        sample_ids += synonym_samples_flattened
        random.shuffle(sample_ids)

    logger.info(
        f"Final Dataset -> {len(sample_ids)}{f' ({samples_number_old} without syn.)' if args.use_synonyms else ''}")

    # create data based on number of samples and transform to tf dataset
    tokens, labels, sample_weights, _ = prepare_token_cls_data(
        sample_numbers=sample_ids,
        other_labels_weight=args.other_labels_weight,
        label_set=args.labels,
        activity_masking=args.activity_masking
    )

    # create and batch tf dataset
    tf_dataset = _create_dataset(tokens["input_ids"], tokens["attention_mask"], labels, sample_weights)
    if args.batch_size:
        tf_dataset = tf_dataset.batch(args.batch_size)

    return tf_dataset


def create_token_cls_dataset_cv(args: argparse.Namespace) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    create the dataset for token classification with huggingface transformers bert like models
    split into kfolds splits to use for cross validation
    :param args: args namespace
    :return: list of tuples (train, dev) as tf.data.Dataset objects
    """
    logger.info(f"Create token classification cv dataset (folds={args.folds} - batch_size={args.batch_size})")
    # load samples to include in dataset
    sample_ids = _get_sample_ids(strategy=args.sampling_strategy)
    random.shuffle(sample_ids)
    logger.info(
        f"Generate token data with params: sampling_strategy={args.sampling_strategy} - use_synonyms={args.use_synonyms}"
        f" - labels={args.labels} - other_labels_weight={args.other_labels_weight}")
    logger.info(f"Basis are {len(sample_ids)} samples from strategy '{args.sampling_strategy}'")

    # create datasets for k fold cross validation
    folded_datasets = []

    kfold = KFold(n_splits=5)
    for i, (train, dev) in enumerate(kfold.split(sample_ids)):

        train_samples = [p for j, p in enumerate(sample_ids) if j in train]
        dev_samples = [p for j, p in enumerate(sample_ids) if j in dev]

        # include synonyms in train samples
        train_samples_number_old = len(train_samples)
        if args.use_synonyms:
            train_synonym_samples = [synonyms for original_sample_id, synonyms in synonyms_of_original_samples.items()
                                     if original_sample_id in train_samples]
            train_synonym_samples_flattened = [item for sublist in train_synonym_samples for item in sublist]
            train_samples += train_synonym_samples_flattened
            random.shuffle(train_samples)

        logger.info(
            f"Fold {i} -> {len(train_samples)}{f' ({train_samples_number_old} without syn.)' if args.use_synonyms else ''}"
            f"/ {len(dev_samples)}")

        # create train data based on number of samples and transform to tf dataset
        tokens, labels, sample_weights, _ = prepare_token_cls_data(
            sample_numbers=train_samples,
            other_labels_weight=args.other_labels_weight,
            label_set=args.labels,
            activity_masking=args.activity_masking
        )
        train_tf_dataset = _create_dataset(tokens["input_ids"], tokens["attention_mask"], labels, sample_weights)

        # create dev data based on number of samples and transform to tf dataset
        tokens, labels, sample_weights, _ = prepare_token_cls_data(
            sample_numbers=dev_samples,
            other_labels_weight=args.other_labels_weight,
            label_set=args.labels,
            activity_masking=args.activity_masking
        )
        dev_tf_dataset = _create_dataset(tokens["input_ids"], tokens["attention_mask"], labels, sample_weights)

        # batch both datasets
        if args.batch_size:
            train_tf_dataset = train_tf_dataset.batch(args.batch_size)
            dev_tf_dataset = dev_tf_dataset.batch(args.batch_size)

        folded_datasets.append((train_tf_dataset, dev_tf_dataset))

    return folded_datasets


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
    parser.add_argument("--labels", default=ALL, type=str, help="Label set to use.")
    parser.add_argument("--other_labels_weight", default=0.1, type=float, help="Sample weight for non gateway tokens.")
    parser.add_argument("--sampling_strategy", default=NORMAL, type=str, help="How to sample samples.")
    parser.add_argument("--use_synonyms", default=True, type=str, help="Include synonym samples.")
    parser.add_argument("--activity_masking", default=NOT, type=str, help="How to include activity data.")

    args_tc = parser.parse_args([] if "__file__" not in globals() else None)

    if True:
        folded_datasets_tc = create_token_cls_dataset_cv(args_tc)
        for i, (train, dev) in enumerate(folded_datasets_tc):
            print(f"Fold {i}: train {len(train)} / dev {len(dev)}")

    if True:
        full_dataset_tc = create_token_cls_dataset_full(args_tc)
        print(f"Full dataset size: {len(full_dataset_tc)}")

