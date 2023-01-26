# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import logging
import argparse
from typing import List, Tuple
import random

from sklearn.model_selection import KFold
import tensorflow as tf
import transformers

from PetReader import pet_reader
from relation_approaches.activity_relation_data_preparation import get_activity_relations
from utils import set_seeds, config, ROOT_DIR, load_pickle, save_as_pickle
from labels import *

logger = logging.getLogger('Data Preparation [Activity Relations]')

# mapping of textual and numerical labels
text_labels = {
    DF: AR_LABEL_DF_FOLLOWING,
    EXCLUSIVE: AR_LABEL_EXCLUSIVE,
    CONCURRENT: AR_LABEL_CONCURRENT,
    NON_RELATED: AR_LABEL_NON_RELATED
}

_tokenizer = transformers.AutoTokenizer.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
assert isinstance(_tokenizer, transformers.PreTrainedTokenizerFast)


def _split_test_set(args: argparse.Namespace, relations: List = None) -> Tuple[List, List]:
    """
    shuffle and split test set from relations
    :param args: args
    :return: data set as list, test set as list
    """
    if not relations:
        relations = get_activity_relations(return_type=dict)
    random.shuffle(relations)
    split_point = int(len(relations) * args.test_share)
    return relations[split_point:], relations[:split_point]


def _create_dataset(input_ids: tf.Tensor, attention_masks: tf.Tensor, labels: tf.Tensor) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))


def _prepare_dataset(relations: List, cache_path: str) -> tf.data.Dataset:
    """
    process relation pairs and related text sections to a dataset
    :param relations: list of relations to include in the dataset
    :param cache_path: path to store and (maybe) reload results
    :return: tensorflow dataset
    """
    # reload from cache if already exists
    if os.path.exists(cache_path):
        tokens, labels = load_pickle(cache_path)
        logger.info("Reloaded activity relation data from cache")
        results = (tokens, labels)

    else:
        texts = []
        activity_tuples = []
        labels = []
        for i, relation in enumerate(relations[:1]):
            texts.append(' '.join([' '.join(s) for i, s in enumerate(pet_reader.get_doc_sentences(relation[DOC_NAME]))
                                  if i in range(relation[ACTIVITY_1][0], relation[ACTIVITY_2][0] + 1)]))
            activity_tuples.append((' '.join(relation[ACTIVITY_1][2]), ' '.join(relation[ACTIVITY_2][2])))
            labels.append(text_labels[relation[RELATION_TYPE]])
        results = (_tokenize_textual_features(texts, activity_tuples), tf.constant(labels))
        logger.info("Save activity relation data to cache")
        save_as_pickle(results, cache_path)

    return _create_dataset(results[0]["input_ids"], results[0]["attention_mask"], results[1])


def _tokenize_textual_features(texts: List[str], activity_tuples: List[Tuple[str, str]]) -> transformers.BatchEncoding:
    """
    tokenize pairs of (text, (activity 1, activity 2))
    :param texts: list of text
    :param activity_tuples: list of activity tuples
    :return: BatchEncoding
    """
    # tokenize text & tuples separately, because it is not possible to concat triple
    max_length = 512
    text_tokens = _tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')
    activity_tuple_tokens = _tokenizer(activity_tuples, padding=True, return_tensors="tf")

    # concat manually after (cut the CLS token of the second pair); activities first because at the end it may be sliced
    # because of over length of the text
    concatted_input_ids = tf.concat([activity_tuple_tokens["input_ids"], text_tokens["input_ids"][:, 1:]], axis=1)
    concatted_attention_masks = tf.concat([activity_tuple_tokens["attention_mask"],
                                           text_tokens["attention_mask"][:, 1:]], axis=1)

    # limit again to max length because concatenated tuple can be longer than 512
    concatted_input_ids = concatted_input_ids[:, :max_length]
    concatted_attention_masks = concatted_attention_masks[:, :max_length]

    return transformers.BatchEncoding({"input_ids": concatted_input_ids, "attention_mask": concatted_attention_masks})


def create_activity_relation_cls_dataset_cv(args: argparse.Namespace) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    create the dataset for relation classification
    split into kfolds splits to use for cross validation
    :param args: args namespace
    :return: list of tuples (train, dev) as tf.data.Dataset objects
    """
    logger.info(
        f"Create activity relation classification cv dataset (folds={args.folds} - batch_size={args.batch_size})")
    rest, test = _split_test_set(args)
    logger.info(f"Basis are {len(rest)} relations (already splitted test = {len(test)})")

    folded_datasets = []
    kfold = KFold(n_splits=args.folds)
    logger.info(f"Split data in {args.folds} folds")

    for i, (train, dev) in enumerate(kfold.split(rest)):
        train_relations = [p for i, p in enumerate(rest) if i in train]
        dev_relations = [p for i, p in enumerate(rest) if i in dev]

        cache_path = os.path.join(ROOT_DIR, f"data/other/data_cache/activity_relation/data_{args.seed_general}_cv_"
                                            f"fold{i}_")

        train_tf_dataset = _prepare_dataset(train_relations, cache_path=cache_path + "train")
        dev_tf_dataset = _prepare_dataset(dev_relations, cache_path=cache_path + "dev")

        logger.info(f"Fold {i} -> train={len(train_tf_dataset)} / dev={len(dev_tf_dataset)}")

        if args.batch_size:
            train_tf_dataset = train_tf_dataset.batch(args.batch_size)
            dev_tf_dataset = dev_tf_dataset.batch(args.batch_size)

        folded_datasets.append((train_tf_dataset, dev_tf_dataset))

    return folded_datasets


def create_activity_relation_cls_dataset_full(args: argparse.Namespace) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    create one training dataset of the whole data with just a test set separated
    :param args: args namespace
    :return: train set, test set
    """
    logger.info(f"Create full activity relation classification dataset (batch_size={args.batch_size})")
    train, test = _split_test_set(args)
    logger.info(f"Final Dataset -> train={len(train)} / test={len(test)}")

    cache_path = os.path.join(ROOT_DIR, f"data/other/data_cache/activity_relation/data_{args.seed_general}_full_")

    train_tf_dataset = _prepare_dataset(train, cache_path=cache_path + "train")
    test_tf_dataset = _prepare_dataset(test, cache_path=cache_path + "test")

    if args.batch_size:
        train_tf_dataset = train_tf_dataset.batch(args.batch_size)
        test_tf_dataset = test_tf_dataset.batch(args.batch_size)

    return train_tf_dataset, test_tf_dataset


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_general", default=42, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--folds", default=5, type=int, help="K folds.")
    parser.add_argument("--test_share", default=0.1, type=float, help="Share of test set")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    set_seeds(args.seed_general, "args - used for dataset split/shuffling")

    # train, test = create_activity_relation_cls_dataset_full(args)
    # for x in train:
    #     print(x)
    #     break

    create_activity_relation_cls_dataset_cv(args)