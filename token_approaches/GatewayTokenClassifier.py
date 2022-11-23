#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.abspath(''), os.pardir))
sys.path.insert(0, parent_dir)

import argparse
import logging
from typing import List

import tensorflow as tf
import transformers

from metrics import *
from utils import config

logger = logging.getLogger('Gateway Token Classifier')


class GatewayTokenClassifier(tf.keras.Model):

    def __init__(self, args: argparse.Namespace, token_cls_model=None, train_size: int = None,
                 weights_path: str = None) -> None:
        """
        creates a GatewayTokenClassifier
        :param args: args Namespace
        :param token_cls_model: bert like transformer token classification model
        :param train_size: train dataset size
        :param weights_path: path of stored weights. If set, load from there
        """
        logger.info("Create and initialize a GatewayTokenClassifier")
        self.weights_path = weights_path

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        }

        # head of the following model is random initialized by the seed.
        #   - in case of single model, seed is set at the beginning of the script
        #   - in case of model in ensemble, seed is set before this constructor call
        if not token_cls_model:
            token_cls_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
                config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME],
                num_labels=config[KEYWORDS_FILTERED_APPROACH][LABEL_NUMBER])

        # includes one dense layer with linear activation function
        predictions = token_cls_model(inputs).logits
        super().__init__(inputs=inputs, outputs=predictions)

        # B) COMPILE (only needed when training is intended)
        if args and train_size:
            optimizer, lr_schedule = transformers.create_optimizer(
                init_lr=2e-5,
                num_train_steps=(train_size // args.batch_size) * args.epochs,
                weight_decay_rate=0.01,
                num_warmup_steps=0,
            )

            self.compile(optimizer=optimizer,
                         # loss=custom_loss,
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         # general accuracy of all labels (except 0 class for padding tokens)
                         weighted_metrics=[tf.metrics.SparseCategoricalAccuracy(name="overall_accuracy")],
                         # metrics for classes of interest
                         metrics=[xor_precision, xor_recall, xor_f1, and_recall, and_precision, and_f1])
            # token_cls_model.summary()
            # self.summary()

        # if model path is passed, restore weights
        if self.weights_path:
            logger.info(f"Restored weights from {weights_path}")
            self.load_weights(weights_path)

    def predict(self, tokens: transformers.BatchEncoding) -> np.ndarray:
        """
        create predictions for given data
        :param tokens: tokens as BatchEncoding
        :return: numpy array of predictions
        """
        return super().predict({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})


def convert_predictions_into_labels(predictions: np.ndarray, word_ids: List[List[int]]) -> List[List[int]]:
    """
    convert predictions for every token (logits) into a list of labels for each sample
    :param predictions: logits as np.ndarray
    :param word_ids: original word ids of tokens
    :return: list of labels for each sample
    """
    converted_results = []  # list (for each sample): a dict with word_id: predicted class(es))

    for i, sample in enumerate(predictions):
        important_token_pairs = [(i, word_id) for i, word_id in enumerate(word_ids[i]) if word_id is not None]
        converted_sample = {}

        # store prediction(s) for every original word
        for token_index, word_id in important_token_pairs:
            token_prediction = np.argmax(sample[token_index])
            if word_id not in converted_sample:
                converted_sample[word_id] = [token_prediction]
            else:
                converted_sample[word_id].append(token_prediction)

        # merge predictions (possible/necessary if multiple BERT-tokens are mapped to one input word)
        for word_id, token_predictions in converted_sample.items():
            token_predictions = list(set(token_predictions))
            # if different labels were predicted, take the highest => 3 (AND) > 2 (XOR) > 1(other)
            if len(token_predictions) > 1:
                token_predictions.sort(reverse=True)
                token_predictions = token_predictions[:1]
            converted_sample[word_id] = token_predictions[0]

        # assure sort by index after extracting from (unordered?) dict
        converted_sample = [(idx, label) for idx, label in converted_sample.items()]
        converted_sample.sort(key=lambda idx_label_pair: idx_label_pair[0])
        # reduce to ordered list of labels
        converted_sample = [label for idx, label in converted_sample]

        converted_results.append(converted_sample)

    return converted_results
