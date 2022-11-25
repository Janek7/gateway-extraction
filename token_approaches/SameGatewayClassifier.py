#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import argparse
import logging

import tensorflow as tf
import transformers
from petreader.labels import XOR_GATEWAY

from token_approaches.same_gateway_data_preparation import create_same_gateway_cls_dataset_full, \
    create_same_gateway_cls_dataset_cv
from training import cross_validation, full_training
from labels import *
from utils import config, generate_args_logdir, set_seeds

logger = logging.getLogger('Same Gateway Classifier')


parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed_general", default=42, type=int, help="Random seed.")
parser.add_argument("--ensemble", default=True, type=bool, help="Use ensemble learning with config.json seeds.")
parser.add_argument("--seeds_ensemble", default="0-1", type=str, help="Random seed range to use for ensembles")
# routine params
parser.add_argument("--routine", default="cv", type=str, help="Simple split training 'sp', cross validation 'cv' or "
                                                              "full training without validation 'ft'.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")
# Architecture / data params
parser.add_argument("--gateway", default=XOR_GATEWAY, type=str, help="Type of gateway to classify")
parser.add_argument("--context_size", default=1, type=int, help="Number of sentences around to include in text.")
parser.add_argument("--mode", default=CONCAT, type=str, help="How to include gateway information.")
parser.add_argument("--n_gram", default=1, type=int, help="Number of tokens to include for gateway in CONCAT mode.")


class SameGatewayClassifier(tf.keras.Model):
    """
    binary classification model to classify if two gateways belong to the same gateway construct
    """

    def __init__(self, args: argparse.Namespace, bert_model, train_size: int = None):
        logger.info("Create and initialize a SameGatewayClassifier")

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "indexes": tf.keras.layers.Input(shape=[2], dtype=tf.int32)
        }

        if not bert_model:
            bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
        self.bert_model = bert_model
        self.dropout1 = tf.keras.layers.Dropout(0.2)

        # extract cls token for every sample

        if args.mode == CONCAT:
            self.predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        elif args.mode == INDEX:
            self.hidden_layer = tf.keras.layers.Dense(32, activation=tf.nn.relu)
            self.dropout2 = tf.keras.layers.Dropout(0.2)
            self.predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        else:
            raise ValueError(f"mode must be {INDEX} or {CONCAT}")

        super().__init__(self, inputs, self.predictions)

        # B) COMPILE (only needed when training is intended)
        optimizer, lr_schedule = transformers.create_optimizer(
            init_lr=2e-5,
            num_train_steps=(train_size // args.batch_size) * args.epochs,
            weight_decay_rate=0.01,
            num_warmup_steps=0,
        )

        self.compile(optimizer=optimizer,
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics=[tf.keras.metrics.BinaryAccuracy(),
                              tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

        # self.summary()

    def call(self, inputs):
        bert_output = self.bert_model({"input_ids": inputs["input_ids"],
                                       "attention_mask": inputs["attention_mask"]}).last_hidden_state
        # extract cls token for every sample
        cls_token = bert_output[:, 0]
        dropout1 = self.dropout1(cls_token)

        if self.mode == CONCAT:
            return self.predictions(dropout1)
        elif self.mode == INDEX:
            indexes = tf.cast(inputs["indexes"], tf.float32)
            concatted = tf.keras.layers.Concatenate()([dropout1, indexes])
            hidden_layer = self.hidden_layer(concatted)
            dropout2 = self.dropout2(hidden_layer)
            return self.predictions(dropout2)


def train_routine(args: argparse.Namespace) -> None:
    """
    run SameGatewayClassifier training based on passed args
    :param args: namespace args
    :return:
    """

    # Load the model
    # logger.info(f"Load transformer model ({config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME]})")
    # bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])

    # cross validation
    if args.routine == 'cv':
        folded_datasets = create_same_gateway_cls_dataset_cv(args.gateway, args, shuffle=True)
        cross_validation(args, SameGatewayClassifier, folded_datasets)

    # full training without validation
    elif args.routine == 'ft':
        train = create_same_gateway_cls_dataset_full(args.gateway, args, shuffle=True)
        full_training(args, SameGatewayClassifier, train)

    else:
        raise ValueError(f"Invalid training routine: {args.routine}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.logdir = generate_args_logdir(args, script_name="SameGatewayClassifier")

    # this seed is used by default (only overwritten in case of ensemble)
    set_seeds(args.seed_general, "args - used for dataset split/shuffling")

    train_routine(args)
