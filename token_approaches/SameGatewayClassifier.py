#!/usr/bin/env python3

import argparse
import logging
import os
import datetime
import re

import tensorflow as tf
import transformers

from labels import *
from utils import config, set_seeds

logger = logging.getLogger('Same Gateway Classifier')
# logger_ensemble = logging.getLogger('Same Gateway Classifier Ensemble')


parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed_general", default=42, type=int, help="Random seed.")
# routine params
parser.add_argument("--routine", default="cv", type=str, help="Simple split training 'sp', cross validation 'cv' or "
                                                              "full training without validation 'ft'.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")


class SameGatewayClassifier(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, mode: str = CONCAT, train_size: int = None):

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "indexes": tf.keras.layers.Input(shape=[2], dtype=tf.int32)
        }

        bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
        # includes one dense layer with linear activation function
        bert_output = bert_model({"input_ids": inputs["input_ids"],
                                  "attention_mask": inputs["attention_mask"]}).last_hidden_state
        # extract cls token for every sample
        cls_token = bert_output[:, 0]
        dropout1 = tf.keras.layers.Dropout(0.2)(cls_token)
        if mode == CONCAT:
            predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout1)
        elif mode == INDEX:
            indexes = tf.cast(inputs["indexes"], tf.float32)
            concatted = tf.keras.layers.Concatenate()([dropout1, indexes])
            hidden_layer = tf.keras.layers.Dense(32, activation=tf.nn.relu)(concatted)
            dropout2 = tf.keras.layers.Dropout(0.2)(hidden_layer)
            predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout2)

        super().__init__(inputs=inputs, outputs=predictions)

        # B) COMPILE (only needed when training is intended)
        optimizer, lr_schedule = transformers.create_optimizer(
            init_lr=2e-5,
            num_train_steps=(train_size // 8) * 1,
            weight_decay_rate=0.01,
            num_warmup_steps=0,
        )

        self.compile(optimizer=optimizer,
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics=[tf.keras.metrics.BinaryAccuracy()])

        self.summary()


def train_routine(args: argparse.Namespace) -> None:
    # Create logdir name
    args.logdir = os.path.join("../data/logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # this seed is used by default (only overwritten in case of ensemble)
    set_seeds(args.seed_general, "args - used for dataset split/shuffling")

    train_routine(args)