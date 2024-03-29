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
from typing import List, Tuple
import re
import json

import tensorflow as tf
import transformers
import numpy as np
from petreader.labels import XOR_GATEWAY

from Ensemble import Ensemble
from token_approaches.same_gateway_data_preparation import create_same_gateway_cls_dataset_full, \
    create_same_gateway_cls_dataset_cv, preprocess_gateway_pair
from training import cross_validation, full_training
from labels import *
from utils import config, generate_args_logdir, set_seeds, NumpyEncoder

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
parser.add_argument("--routine", default="cv", type=str, help="Cross validation 'cv' or "
                                                              "full training without validation 'ft'.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")
# Data params
parser.add_argument("--gateway", default=XOR_GATEWAY, type=str, help="Type of gateway to classify")
parser.add_argument("--use_synonyms", default=False, type=str, help="Include synonym samples.")
parser.add_argument("--context_size", default=1, type=int, help="Number of sentences around to include in text.")
parser.add_argument("--mode", default=CONTEXT_TEXT_AND_LABELS_NGRAM, type=str, help="Architecture variant.")
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
    def __init__(self, args: argparse.Namespace = None, bert_model=None, train_size: int = None,
                 weights_path: str = None) -> None:
        """
        creates a SameGatewayClassifier
        :param args: args Namespace
        :param bert_model: bert like transformer model
        :param train_size: train dataset size
        :param weights_path: path of stored weights. If set, load from there
        """
        logger.info("Create and initialize a SameGatewayClassifier")
        if not args:
            logger.warning("No parsed args passed to SameGatewayClassifier, use standard values")
            args = parser.parse_args([] if "__file__" not in globals() else None)
        self.args = args
        self.weights_path = weights_path

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "indexes": tf.keras.layers.Input(shape=[2], dtype=tf.int32),
            "context_labels": tf.keras.layers.Input(shape=[config[SAME_GATEWAY_CLASSIFIER][CONTEXT_LABEL_LENGTH]],
                                                    dtype=tf.int32),
        }

        if not bert_model:
            bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
        # includes one dense layer with linear activation function
        bert_output = bert_model({"input_ids": inputs["input_ids"],
                                  "attention_mask": inputs["attention_mask"]}).last_hidden_state
        # extract cls token for every sample
        cls_token = bert_output[:, 0]
        dropout1 = tf.keras.layers.Dropout(self.args.dropout)(cls_token)

        # for only textual modes add immediately output layers
        if args.mode == CONTEXT_NGRAM or args.mode == N_GRAM:
            predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout1)

        # for modes that include more features, combine them with hidden layer(s) with BERT output
        elif args.mode in [CONTEXT_INDEX, CONTEXT_LABELS_NGRAM, CONTEXT_TEXT_AND_LABELS_NGRAM]:
            if args.mode == CONTEXT_INDEX:
                additional_information = inputs["indexes"]
            elif args.mode in [CONTEXT_LABELS_NGRAM, CONTEXT_TEXT_AND_LABELS_NGRAM]:
                additional_information = inputs["context_labels"]
            additional_information = tf.cast(additional_information, tf.float32)
            hidden = tf.keras.layers.Concatenate()([dropout1, additional_information])
            for hidden_layer_size in args.hidden_layer.split("-"):
                hidden = tf.keras.layers.Dense(int(hidden_layer_size), activation=tf.nn.relu)(hidden)
                hidden = tf.keras.layers.Dropout(args.dropout)(hidden)
            predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)

        else:
            raise ValueError(f"mode must be {N_GRAM}, {CONTEXT_INDEX}, {CONTEXT_NGRAM}, {CONTEXT_LABELS_NGRAM} or"
                             f" {CONTEXT_TEXT_AND_LABELS_NGRAM}")

        super().__init__(inputs=inputs, outputs=predictions)

        # B) COMPILE (only needed when training is intended)
        if args and train_size:
            logger.info("Create optimizer for training")
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

        # if model path is passed, restore weights
        if self.weights_path:
            logger.info(f"Restored weights from {weights_path}")
            self.load_weights(weights_path).expect_partial()

    def classify_pair(self, doc_name, g1, g2) -> np.float32:
        """
        create prediction for given data as number
        :param doc_name: document where gateways belong to
        :param g1: first gateway of pair to evaluate
        :param g2: second gateway of pair to evaluate
        :return: true or false (threshold 0.5 because of binary classification head)
        """
        # preprocess data
        tokens, indexes, context_labels = preprocess_gateway_pair(self.args, doc_name, g1, g2)
        inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "indexes": indexes,
            "context_labels": context_labels
        }
        return super().predict(inputs)[0][0]

    def classify_pair_bool(self, doc_name, g1, g2) -> bool:
        """ create prediction for given data as number """
        return self.classify_pair(doc_name, g1, g2) > 0.5


class SGCEnsemble(Ensemble):
    """
    Ensemble (seeds) of same gateway classification model
    - During training normal Ensemble class is used
    - Used only for inference -> extends normal Ensemble by classification methods
    """

    def __init__(self, log_folder: str, seeds: List = None, ensemble_path: str = None, es_monitor: str = 'val_loss',
                 seed_limit: int = None, **kwargs) -> None:
        """
        see super class for param description
        override for fixing model class
        :param log_folder: log_folder where to store results
        """
        self.log_folder = log_folder
        self.predictions = {}
        # in case of reload of ensemble args are not passed -> create args, extract used mode from path and set
        if ensemble_path:
            logger.info("Use standard values of args when reloading ensemble")
            args = parser.parse_args([] if "__file__" not in globals() else None)
            mode_pattern = re.compile(",m=([a-z_]+)")
            args.mode = mode_pattern.search(ensemble_path).group(1)
            logger.info(f"Reload model with mode {args.mode}")
            kwargs["args"] = args

        super().__init__(SameGatewayClassifier, seeds, ensemble_path, es_monitor, seed_limit, **kwargs)

    def classify_pair(self, doc_name: str, g1: Tuple, g2) -> np.float32:
        """
        create predictions for given data with each model as number
        :param doc_name: document where gateways belong to
        :param g1: first gateway of pair to evaluate
        :param g2: second gateway of pair to evaluate
        :return: true or false (threshold 0.5 because of binary classification head)
        """
        predictions = [model.classify_pair(doc_name, g1, g2) for model in self.models]
        predictions_averaged = np.mean(predictions, axis=0)
        self.log_prediction(doc_name, g1, g2, predictions_averaged, predictions)
        return predictions_averaged

    def classify_pair_bool(self, doc_name, g1, g2) -> bool:
        """ create prediction for given data as number """
        return self.classify_pair(doc_name, g1, g2) > 0.5

    def log_prediction(self, doc_name, g1, g2, predictions_averaged, predictions, comment="normal"):
        """
        log prediction of two gateways into internal log
        """
        if doc_name not in self.predictions:
            self.predictions[doc_name] = []
        self.predictions[doc_name].append({"gateway_1": g1, "gateway_2": g2, "label": int(predictions_averaged > 0.5),
                                           "predictions_averaged": predictions_averaged, "predictions": predictions,
                                           "comment": comment})

    def save_prediction_logs(self) -> None:
        """
        save predictions dictionary to json file in output_folder of approach
        :return:
        """
        path = os.path.join(self.log_folder, "sg_classifications.json")
        logger.info(f"Save prediction results into {path}")
        with open(path, 'w') as file:
            json.dump(self.predictions, file, indent=4, cls=NumpyEncoder)


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
        folded_datasets = create_same_gateway_cls_dataset_cv(args.gateway, args)
        cross_validation(args, SameGatewayClassifier, folded_datasets)

    # full training without validation
    elif args.routine == 'ft':
        train = create_same_gateway_cls_dataset_full(args.gateway, args)
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

    # sgc = SameGatewayClassifier(weights_path='C:\\Users\\janek\\Development\\Git\\master-thesis\\data\\logs_server\\_final\\ensemble_,m=context_text_and_labels_n_gram,se=10-11\\10\\weights\\weights')
    #
    # sgce = SGCEnsemble(ensemble_path="C:\\Users\\janek\\Development\\Git\\master-thesis\\data\\logs_server\\_final\\ensemble_,m=context_text_and_labels_n_gram,se=10-11")
