#!/usr/bin/env python3

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

import tensorflow as tf
import transformers
from petreader.labels import XOR_GATEWAY

from token_approaches.same_gateway_data_preparation import create_same_gateway_cls_dataset_full, \
    create_same_gateway_cls_dataset_cv
from training import cross_validation, full_training
from labels import *
from utils import config, generate_args_logdir, set_seeds

logger = logging.getLogger('Same Gateway Classifier')
logger_ensemble = logging.getLogger('Same Gateway Classifier Ensemble')

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
# Data params
parser.add_argument("--gateway", default=XOR_GATEWAY, type=str, help="Type of gateway to classify")
parser.add_argument("--use_synonyms", default=False, type=str, help="Include synonym samples.")
parser.add_argument("--context_size", default=1, type=int, help="Number of sentences around to include in text.")
parser.add_argument("--mode", default=CONTEXT_NGRAM, type=str, help="How to include gateway information.")
parser.add_argument("--n_gram", default=1, type=int, help="Number of tokens to include for gateway in CONCAT mode.")
parser.add_argument("--activity_masking", default=NOT, type=str, help="How to include activity data.")
# Architecture params
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")
parser.add_argument("--hidden_layer", default="32", type=str, help="Hidden layer sizes sep. by '-'")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--warmup", default=0, type=int, help="Number of warmup steps.")


class SameGatewayClassifier(tf.keras.Model):
    """
    binary classification model to classify if two gateways belong to the same gateway construct
    """
    def __init__(self, args: argparse.Namespace, bert_model, train_size: int = None):

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "indexes": tf.keras.layers.Input(shape=[2], dtype=tf.int32),
            "context_labels": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
        }

        if not bert_model:
            bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
        # includes one dense layer with linear activation function
        bert_output = bert_model({"input_ids": inputs["input_ids"],
                                  "attention_mask": inputs["attention_mask"]}).last_hidden_state
        # extract cls token for every sample
        cls_token = bert_output[:, 0]
        dropout1 = tf.keras.layers.Dropout(args.dropout)(cls_token)

        # for only textual modes add immediately output layers
        if args.mode == CONTEXT_NGRAM or args.mode == N_GRAM:
            predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout1)

        # for modes that include more features, combine them with hidden layer(s) with BERT output
        elif args.mode == CONTEXT_INDEX or CONTEXT_LABELS_NGRAM:
            if args.mode == CONTEXT_INDEX:
                additional_information = inputs["indexes"]
            elif args.mode == CONTEXT_LABELS_NGRAM:
                additional_information = inputs["context_labels"]
            additional_information = tf.cast(additional_information, tf.float32)
            hidden = tf.keras.layers.Concatenate()([dropout1, additional_information])
            for hidden_layer_size in args.hidden_layer.split("-"):
                hidden = tf.keras.layers.Dense(int(hidden_layer_size), activation=tf.nn.relu)(hidden)
                hidden = tf.keras.layers.Dropout(args.dropout)(hidden)
            predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)

        else:
            raise ValueError(f"mode must be {N_GRAM}, {CONTEXT_NGRAM}, {CONTEXT_INDEX} or {CONTEXT_LABELS_NGRAM}")

        super().__init__(inputs=inputs, outputs=predictions)

        # B) COMPILE (only needed when training is intended)
        optimizer, lr_schedule = transformers.create_optimizer(
            init_lr=args.learning_rate,
            num_train_steps=(train_size // args.batch_size) * args.epochs,
            weight_decay_rate=0.01,
            num_warmup_steps=args.warmup,
        )

        self.compile(optimizer=optimizer,
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics=[tf.keras.metrics.BinaryAccuracy(),
                              tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

        # self.summary()


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
