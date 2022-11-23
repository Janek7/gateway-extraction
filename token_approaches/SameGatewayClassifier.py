#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.abspath(''), os.pardir))
sys.path.insert(0, parent_dir)

import argparse
import logging

import tensorflow as tf
import transformers

from labels import *
from utils import config

logger = logging.getLogger('Same Gateway Classifier')
logger_ensemble = logging.getLogger('Same Gateway Classifier Ensemble')


class SameGatewayClassifier(tf.keras.Model):
    """
    binary classification model to classify if two gateways belong to the same gateway construct
    """
    def __init__(self, args: argparse.Namespace, bert_model, mode: str = CONCAT, train_size: int = None):

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "indexes": tf.keras.layers.Input(shape=[2], dtype=tf.int32)
        }

        if not bert_model:
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
        else:
            raise ValueError(f"mode must be {INDEX} or {CONCAT}")

        super().__init__(inputs=inputs, outputs=predictions)

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

        self.summary()
