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
    def __init__(self, args: argparse.Namespace, bert_model=None, train_size: int = None) -> None:
        """
        creates a SameGatewayClassifier
        :param args: args Namespace
        :param bert_model: bert like transformer model
        :param train_size: train dataset size
        """
        logger.info("Create and initialize a SameGatewayClassifier")

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        }

        if not bert_model:
            token_cls_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
                config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME],
                num_labels=config[KEYWORDS_FILTERED_APPROACH][LABEL_NUMBER])

        bert_output = bert_model(inputs).logits
        super().__init__(inputs=inputs, outputs=bert_output)

        # B) COMPILE (only needed when training is intended)
        if args and train_size:
            optimizer, lr_schedule = transformers.create_optimizer(
                init_lr=2e-5,
                num_train_steps=(train_size // args.batch_size) * args.epochs,
                weight_decay_rate=0.01,
                num_warmup_steps=0,
            )

            self.compile(optimizer=optimizer,
                         loss=tf.keras.losses.SparseCategoricalCrossentropy())
            # token_cls_model.summary()
            # self.summary()


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