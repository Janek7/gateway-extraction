import logging
from typing import Tuple, List

import tensorflow as tf
import transformers
from sklearn.model_selection import KFold
from transformers import BatchEncoding
from PetReader import pet_reader
from petreader.labels import *

from labels import *
from utils import config

logger = logging.getLogger('Data Preparation')


tokenizer = transformers.AutoTokenizer.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def preprocess_tokenization_data(other_labels_weight: float = 0.1, label_set: str = 'filtered',
                                 sample_numbers: List[int] = None)\
        -> Tuple[BatchEncoding, tf.Tensor, tf.Tensor, List[List[int]]]:
    """
    create token classification samples from whole PET dataset -> samples (tokens) and their labels and weights for
    usage in a tensorflow dataset
    :param other_labels_weight: sample weight to assign samples with tokens != gateway tokens
    :param label_set: flag if to use all labels ('all') or only gateway labels and one rest label ('filtered')
    :param sample_numbers: list of sample IDs to preprocess; if None -> use all
    :return: tokens, labels & weights as tensors, original word ids (2-dim integer list)
    """
    if not sample_numbers:
        sample_numbers = pet_reader.token_dataset.GetRandomizedSampleNumbers()
    sample_dicts = [pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number) for sample_number in
                    sample_numbers]
    sample_sentences = [sample_dict['tokens'] for sample_dict in sample_dicts]

    # 1) transform tokens tags into IDs classification
    dataset_tokens = tokenizer(sample_sentences, is_split_into_words=True, padding=True, return_tensors='tf')
    max_sentence_length = dataset_tokens['input_ids'].shape[1]

    # 2) transform NER token tags into labels for classification
    dataset_labels = []
    dataset_sample_weights = []
    dataset_word_ids = []
    for i, sample_number in enumerate(sample_numbers):
        sample_dict = pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number)
        # transformer_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][i])
        # tokenize again every single sample to get access to .word_ids()
        tokenization = tokenizer(sample_dict['tokens'], is_split_into_words=True,
                                 padding='max_length', max_length=max_sentence_length, return_tensors='tf')
        sample_tokens = tokenizer.convert_ids_to_tokens(tokenization['input_ids'][0])

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


def shuffle_tokenization_data(input_ids: tf.Tensor, attention_masks: tf.Tensor, labels: tf.Tensor,
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


def create_dataset(input_ids: tf.Tensor, attention_masks: tf.Tensor, labels: tf.Tensor, sample_weights: tf.Tensor)\
        -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks},
                                               labels,
                                               sample_weights))


def create_full_training_dataset(other_labels_weight: float, label_set: str = 'filtered', batch_size: int = None,
                                 shuffle: bool = True) -> tf.data.Dataset:
    """
    create one training set of the whole data without separating a dev set
    :param other_labels_weight: sample weight to assign samples with tokens != gateway tokens
    :param label_set: flag if to use all labels ('all') or only gateway labels and one rest label ('filtered')
    :param batch_size: apply batching to size if given
    :param shuffle: flag if shuffle the data
    :return: one tensorflow dataset
    """
    logger.info(f"Create full training dataset dataset (other_labels_weight: {other_labels_weight} - "
                f"label_set: {label_set} - batch_size: {batch_size} - shuffle: {shuffle})")
    tokens, labels, sample_weights, _ = preprocess_tokenization_data(other_labels_weight=other_labels_weight,
                                                                     label_set=label_set)
    dataset = create_dataset(tokens["input_ids"], tokens["attention_mask"], labels, sample_weights)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def create_token_classification_dataset_cv(other_labels_weight: float, label_set: str = 'filtered', kfolds: int = 5,
                                           batch_size: int = None, shuffle: bool = True)\
        -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    create the dataset for token classification with huggingface transformers bert like models
    split into kfolds splits to use for cross validation
    :param other_labels_weight: sample weight to assign samples with tokens != gateway tokens
    :param label_set: flag if to use all labels ('all') or only gateway labels and one rest label ('filtered')
    :param kfolds: number of folds
    :param batch_size: apply batching to size if given
    :param shuffle: flag if shuffle the data
    :return: list of tuples (train, dev) as tf.data.Dataset objects
    """
    logger.info(f"Create CV (folds={kfolds}) dataset (other_labels_weight: {float} - label_set: {label_set} "
                f"- batch_size: {batch_size} - shuffle: {shuffle})")
    tokens, labels, sample_weights, _ = preprocess_tokenization_data(other_labels_weight=other_labels_weight,
                                                                     label_set=label_set)
    input_ids, attention_masks = tokens['input_ids'], tokens['attention_mask']

    # shuffle inputs before splitting in train/dev
    if shuffle:
        input_ids, attention_masks, labels, sample_weights = shuffle_tokenization_data(input_ids, attention_masks,
                                                                                       labels, sample_weights)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kfolds)

    # create folds
    folded_datasets = []
    for train, test in kfold.split(input_ids):
        train_tf_dataset = create_dataset(tf.gather(input_ids, train),
                                          tf.gather(attention_masks, train),
                                          tf.gather(labels, train),
                                          tf.gather(sample_weights, train))
        dev_tf_dataset = create_dataset(tf.gather(input_ids, test),
                                        tf.gather(attention_masks, test),
                                        tf.gather(labels, test),
                                        tf.gather(sample_weights, test))
        if batch_size:
            train_tf_dataset = train_tf_dataset.batch(batch_size)
            dev_tf_dataset = dev_tf_dataset.batch(batch_size)
        folded_datasets.append((train_tf_dataset, dev_tf_dataset))

    return folded_datasets


def create_token_classification_dataset(other_labels_weight: float, label_set: str = 'filtered', dev_share: float = 0.1,
                                        batch_size: int = None, shuffle: bool = True) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    create the dataset for token classification with huggingface transformers bert like models
    split into train and dev/validation by the given share in dev_share
    tokens are labeled into XOR, AND and OTHER. Additionally a label for bert specific tokens such as CLS, SEP and PAD
    :param other_labels_weight: sample weight to assign samples with tokens != gateway tokens
    :param label_set: flag if to use all labels ('all') or only gateway labels and one rest label ('filtered')
    :param dev_share: share of validation/development dataset
    :param batch_size: apply batching to size if given
    :param shuffle: flag if shuffle the data
    :return: train dataset as tf.data.Dataset, dev dataset as tf.data.Dataset
    """
    logger.info(f"Create simple split (dev_share={dev_share}) dataset ((other_labels_weight: {float} "
                f"- label_set: {label_set} - batch_size: {batch_size} - shuffle: {shuffle})")
    tokens, labels, sample_weights, _ = preprocess_tokenization_data(other_labels_weight=other_labels_weight,
                                                                     label_set=label_set)

    # create tensorflow dataset and split into train and dev
    dataset = create_dataset(tokens["input_ids"], tokens["attention_mask"], labels, sample_weights)
    number_samples = tokens['input_ids'].shape[0]
    val_samples = round(number_samples * dev_share)
    val_dataset = dataset.take(val_samples)
    train_dataset = dataset.skip(val_samples)

    logger.info(f"Created token classification dataset of shape {tokens['input_ids'].shape} splitted into "
                f"{1 - dev_share}/{dev_share} -> {number_samples - val_samples}/{val_samples} (train/dev)")
    if batch_size:
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        logger.info(f"Batch datasets (size {batch_size}) -> {len(train_dataset)}/{len(val_dataset)}")

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=10000)

    return train_dataset, val_dataset


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if False:
        train, dev = create_token_classification_dataset(other_labels_weight=.2, label_set='filtered',
                                                         dev_share=.1, batch_size=8)

    if True:
        folded_datasets = create_token_classification_dataset_cv(other_labels_weight=.2, kfolds=5,
                                                                 label_set='filtered', batch_size=8)
        for t, d in folded_datasets:
            print(len(t), len(d))
