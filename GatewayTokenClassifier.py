#!/usr/bin/env python3

import argparse
import logging
import os
import re
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from metrics import *
from utils import config, set_seeds

logger = logging.getLogger('Gateway Token Classifier')
logger_ensemble = logging.getLogger('Gateway Token Classifier Ensemble')


class GatewayTokenClassifierEnsemble:

    def __init__(self, args: argparse.Namespace = None, token_cls_model = None, train_size: int = None,
                 seeds: List = None, ensemble_path: str = None) -> None:
        """
        initializes a ensemble of GatewayTokenClassifier
        :param args: args Namespace
        :param token_cls_model: bert like transformer token classification model
        :param train_size: train dataset size
        :param seeds: list of seeds for which to create models (default: config seeds)
        :param ensemble_path: path of trained ensemble with stored weights. If set, load model weights from there
        """
        logger_ensemble.info("Create and initialize a GatewayTokenClassifierEnsemble")
        if seeds:
            self.seeds = seeds
        elif not seeds and ensemble_path:
            seed_range_pattern = re.compile("se=(\d+)-(\d+)")
            result = seed_range_pattern.search(ensemble_path)
            self.seeds = list(range(int(result.group(1)), int(result.group(2))))
        self.ensemble_path = ensemble_path
        self.models = []

        # create single models based on seeds or reload from saved weights
        for i, seed in enumerate(self.seeds):
            # set only if weights are not loaded from path
            if not self.ensemble_path:
                set_seeds(seed, "GatewayTokenClassifierEnsemble - model initialization")
            model = GatewayTokenClassifier(args=args, token_cls_model=token_cls_model, train_size=train_size)
            # if path to trained ensemble is passed, restore weights
            if self.ensemble_path:
                path = os.path.join(self.ensemble_path, str(seed), "weights/weights")
                model.load_weights(path).expect_partial()
                logger_ensemble.info(f"Restored weights from trained ensemble {path}")
            self.models.append(model)

    def fit(self, args, train_dataset, dev_dataset=None, save_single_models=False, fold=None):
        """
        fit method that fits every single seed model and averages metrics in history
        :param args: namespace args
        :param train_dataset: train dataset
        :param dev_dataset: dev dataset
        :param save_single_models: if True, record training with Tensorboard and save model weights in subfolder
        :param fold: fold number
        :return: averaged history
        """
        if self.ensemble_path:
            logger.warning("Ensemble was loaded from stored weights and should not be trained further (optimizer not"
                           "was not saved")
        args_logdir_original = args.logdir
        histories = []

        for i, (model, seed) in enumerate(zip(self.models, self.seeds)):
            logger_ensemble.info(f" Fold {fold}: Fit Model {i} with seed {self.seeds[i]} ".center(50, '*'))
            set_seeds(seed, "GatewayTokenClassifierEnsemble - model fit")

            callbacks = []  # [tf.keras.callbacks.EarlyStopping(monitor='overall_accuracy', min_delta=1e-4, patience=1, verbose=0, mode="max", restore_best_weights=True)]

            if save_single_models:
                args.logdir = f"{args_logdir_original}/{seed}"
                os.makedirs(args.logdir, exist_ok=True)
                if i < 5:
                    callbacks.append(tf.keras.callbacks.TensorBoard(args.logdir, update_freq='batch', profile_batch=0))

            history = model.fit(train_dataset, epochs=args.epochs, validation_data=dev_dataset, callbacks=callbacks)
            histories.append(history)

            if save_single_models:
                model.save_weights(os.path.join(args.logdir, "weights/weights"))

        return self.average_histories(histories)

    @staticmethod
    def average_histories(histories: List[tf.keras.callbacks.History]) -> tf.keras.callbacks.History:
        """
        average histories (epoch-wise)
        :param histories: list of tf.keras.callbacks.History (one for each seed)
        :return: merged history as one object (tf.keras.callbacks.History)
        """
        history_merged = tf.keras.callbacks.History()
        for metric in histories[0].history.keys():
            # with axis=0 keep epoch dimensions, just reduce seed dimension
            seed_means = list(np.mean([h.history[metric] for h in histories], axis=0))
            history_merged.history[metric] = seed_means
            # record last epoch value for each seed as well
            history_merged.history[f"seed-results-{metric}"] = [h.history[metric][-1] for h in histories]
        return history_merged

    def predict(self, tokens: transformers.BatchEncoding) -> np.ndarray:
        """
        create predictions for given data with each model and average results on token axis
        :param tokens: tokens as BatchEncoding
        :return: numpy array of averaged predictions
        """
        predictions = [model.predict(tokens=tokens) for model in self.models]
        predictions_averaged = np.mean(predictions, axis=0)
        return predictions_averaged

    def predict_converted(self, tokens: transformers.BatchEncoding, word_ids: List[List[int]]) \
            -> List[List[int]]:
        """
        create averaged predictions for given data and converts outputs into labels
        -> output is one (numerical) label for each input token
        :param tokens: tokens as BatchEncoding
        :param word_ids: original word ids of tokens as 2-dim list
        :return: list of token labels for each sample (two-dim list)
        """
        predictions = self.predict(tokens)
        return convert_predictions_into_labels(predictions, word_ids)


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

    def predict_converted(self, tokens: transformers.BatchEncoding, word_ids: List[List[int]]) \
            -> List[List[int]]:
        """
        create predictions for given data and converts outputs into labels
        -> output is one (numerical) label for each input token
        :param tokens: tokens as BatchEncoding
        :param word_ids: original word ids of tokens as 2-dim list
        :return: list of token labels for each sample (two-dim list)
        """
        predictions = self.predict(tokens)
        return convert_predictions_into_labels(predictions, word_ids)


# shared helper method


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
