#!/usr/bin/env python3

import argparse
import json
import os
import datetime
import re
import logging
from typing import List, Dict

import numpy as np
import tensorflow as tf
import transformers

from token_data_preparation import create_token_classification_dataset, create_token_classification_dataset_cv
from metrics import *
from utils import config, set_seeds

logger = logging.getLogger('Gateway Token Classifier')
logger_ensemble = logging.getLogger('Gateway Token Classifier Ensemble')

parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# routine params
parser.add_argument("--routine", default="cv", type=str, help="Simple training or cross validation.")
parser.add_argument("--dev_share", default=0.1, type=float, help="Share of dev dataset in simple training routine.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")
# Architecture params
parser.add_argument("--ensemble", default=True, type=bool, help="Use ensemble learning with config.json seeds.")
parser.add_argument("--labels", default="filtered", type=str, help="Label set to use.")
parser.add_argument("--other_labels_weight", default=0.1, type=float, help="Sample weight for non gateway tokens.")


class GatewayTokenClassifierEnsemble:

    def __init__(self, args: argparse.Namespace = None, bert_model=None, train_size: int = None,
                 seeds=None) -> None:
        logger_ensemble.info("Create and initialize a GatewayTokenClassifierEnsemble")
        if seeds is None:
            seeds = config[ENSEMBLE_SEEDS]
        self.seeds = seeds
        self.models = []
        # create single models based on seeds
        for i, seed in enumerate(self.seeds):
            set_seeds(seed, "GatewayTokenClassifierEnsemble - model initialization")
            self.models.append(GatewayTokenClassifier(args=args, bert_model=bert_model, train_size=train_size))

    def fit(self, args, fold_seed_datasets):
        """
        fit method that fits every single seed model and averages metrics in history
        :param fold_seed_datasets: list of train/dev pairs (differ in seed; but same fold)
        :return: averaged history
        """
        histories = []

        for i, (model, seed, (train, dev)) in enumerate(zip(self.models, self.seeds, fold_seed_datasets)):
            logger_ensemble.info(f" Fit Model {i} with seed {self.seeds[i]} ".center(50, '*'))
            set_seeds(seed, "GatewayTokenClassifierEnsemble - model fit")

            history = model.fit(
                train, epochs=args.epochs, validation_data=dev,
                callbacks=[
                    # tf.keras.callbacks.TensorBoard(args.logdir, update_freq='batch', profile_batch=0),
                    tf.keras.callbacks.EarlyStopping(monitor='val_overall_accuracy', min_delta=1e-4, patience=1,
                                                     verbose=0, mode="max", restore_best_weights=True)
                ]
            )
            histories.append(history)

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

    def __init__(self, args: argparse.Namespace = None, bert_model=None, train_size: int = None,
                 weights_path: str = None) -> None:
        """
        creates a GatewayTokenClassifier
        :param args: args Namespace
        :param bert_model: bert like transformer for token classification model
        :param train_size: train dataset size
        :param weights_path: path of stored weights. If set, load from there
        """
        logger.info("Create and initialize a GatewayTokenClassifier")
        self.weights_path = weights_path
        num_labels = args.num_labels if args else config[KEYWORDS_FILTERED_APPROACH][NUM_LABELS]

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        }
        if not bert_model:
            bert_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
                config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME], num_labels=num_labels)
        predictions = bert_model(inputs).logits  # includes one dense layer with linear activation function
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


# HELPER METHODS


def convert_predictions_into_labels(predictions: np.ndarray, word_ids: List[List[int]]) -> List[List[int]]:
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


def simple_training(args: argparse.Namespace, token_cls_model) -> None:
    """
    run a training based on a simple train / test split
    :param args: namespace args
    :param token_cls_model: token classification model
    :return:
    """
    logger.info(f"Run simple training (num_labels={args.num_labels}; other_labels_weight={args.other_labels_weight}; "
                f"dev_share={args.dev_share})")
    train, dev = create_token_classification_dataset(other_labels_weight=args.other_labels_weight,
                                                     label_set=args.labels, dev_share=args.dev_share,
                                                     batch_size=args.batch_size)

    # Create the model and train it
    model = GatewayTokenClassifier(args, token_cls_model, len(train))

    history = model.fit(
        train, epochs=args.epochs, validation_data=dev,
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq="batch", profile_batch=0),
                   tf.keras.callbacks.EarlyStopping(monitor='val_overall_accuracy', min_delta=1e-4, patience=1,
                                                    verbose=0, mode="max", restore_best_weights=True)
                   ]
    )

    # store model
    if args.store_weights:
        model.save_weights(os.path.join(args.logdir, "weigths/weights"))

    # store metrics
    with open(os.path.join(args.logdir, "metrics.json"), 'w') as file:
        json.dump(history.history, file, indent=4)


def cross_validation(args: argparse.Namespace, token_cls_model) -> None:
    """
    run training in a cross validation routine -> averaged results are outputted into logdir
    :param args: namespace args
    :param token_cls_model: token classification model
    :return:
    """
    logger.info(f"Run {args.folds}-fold cross validation (num_labels={args.num_labels}; "
                f"other_labels_weight={args.other_labels_weight})")

    # laod data multiple times when using ensembles because seed influences shuffling
    if not args.ensemble:
        folded_datasets = create_token_classification_dataset_cv(other_labels_weight=args.other_labels_weight,
                                                                 label_set=args.labels, kfolds=args.folds,
                                                                 batch_size=args.batch_size)
    else:
        seed_dataset_lists = []
        for seed in config[ENSEMBLE_SEEDS]:
            set_seeds(seed, "GatewayTokenClassifierEnsemble - dataset creation")
            seed_dataset_lists.append(
                create_token_classification_dataset_cv(other_labels_weight=args.other_labels_weight,
                                                       label_set=args.labels, kfolds=args.folds,
                                                       batch_size=args.batch_size))

    os.makedirs(args.logdir, exist_ok=True)
    args_logdir_original = args.logdir

    # models = []  # not used because of memory limitations
    metrics_per_fold = {'avg_val_loss': 0, 'avg_val_xor_precision': 0, 'avg_val_xor_recall': 0, 'avg_val_xor_f1': 0,
                        'avg_val_and_recall': 0, 'avg_val_and_precision': 0, 'avg_val_and_f1': 0, 'avg_val_overall_accuracy': 0,
                        'val_loss': [], 'val_xor_precision': [], 'val_xor_recall': [], 'val_xor_f1': [], 'val_and_recall': [],
                        'val_and_precision': [], 'val_and_f1': [], 'val_overall_accuracy': []}

    def update_avg_metrics(metrics_per_fold):
        for metric, value in metrics_per_fold.items():
            if not metric.startswith("avg_") and not metric.startswith("seed-results-"):
                metrics_per_fold[f"avg_{metric}"] = round(np.mean(value), 4)

    def print_metrics(metrics_per_fold):
        print(' Score per fold '.center(100, '-'))
        for i in range(args.folds):
            if i > 0: print('-' * 100)
            metric_str = ' - '.join([f"{metric}: {round(value[i], 4)}" for metric, value in metrics_per_fold.items() if
                                     not metric.startswith("avg_") and not metric.startswith("seed-results-")])
            print(f"> Fold {i + 1} - {metric_str}")
        print()
        print(' Average scores '.center(100, '-'))
        print(' - '.join([f"{metric}: {round(value, 4)}" for metric, value in metrics_per_fold.items() if
                          metric.startswith("avg_")]))

    # perform k-fold CV

    for i in range(args.folds):
        logger.info(f" Start training of fold {i} ".center(100, '-'))

        # create fold folder
        if not args.ensemble:
            args.logdir = f"{args_logdir_original}/{i + 1}"
            os.makedirs(args.logdir, exist_ok=True)

        # a) fit normal models
        if not args.ensemble:
            train_dataset, dev_dataset = folded_datasets[i][0], folded_datasets[i][1]
            model = GatewayTokenClassifier(args, token_cls_model, len(train_dataset))
            history = model.fit(
                train_dataset, epochs=args.epochs, validation_data=dev_dataset,
                callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq='batch', profile_batch=0),
                           tf.keras.callbacks.EarlyStopping(monitor='val_overall_accuracy', min_delta=1e-4, patience=1,
                                                            verbose=0, mode="max", restore_best_weights=True)
                           ]
            )
            # store model
            if args.store_weights:
                model.save_weights(os.path.join(args.logdir, "weights/weights"))

        # b) fit ensemble model (train multiple seeds for current fold)
        else:
            # extract fold i train/dev pair for the current seed and pass this list to fit of ensemble classifier
            fold_i_seed_datasets = [seed_folded_datasets[i] for seed_folded_datasets in seed_dataset_lists]
            ensemble_model = GatewayTokenClassifierEnsemble(args, token_cls_model,
                                                            train_size=len(fold_i_seed_datasets[0][0]))
            history = ensemble_model.fit(args, fold_i_seed_datasets)

        # record fold results (record only validation results; drop training metrics)
        for metric, values in history.history.items():
            # record value of last epoch for each metric
            if metric.startswith("val_"):
                # values = values of metric in each epoch (in case of ensemble, values is already averaged over seeds)
                metrics_per_fold[metric].append(round(values[-1], 4))
            elif metric.startswith("seed-results-val_"):
                # values = list of results of metric for every seed (last epoch)
                metrics_per_fold[f"{metric}-{i}"] = values

        # models.append(model)

    logger.info("Finished CV, average, print and save results")
    update_avg_metrics(metrics_per_fold)
    print_metrics(metrics_per_fold)

    # save metrics
    with open(os.path.join(args_logdir_original, "cv_metrics.json"), 'w') as file:
        json.dump(metrics_per_fold, file, indent=4)


def train_routine(args: argparse.Namespace) -> None:
    # Create logdir name
    args.logdir = os.path.join("data/logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    if args.labels == 'filtered':
        args.num_labels = 4
    elif args.labels == 'all':
        args.num_labels = 9
    else:
        raise ValueError(f"args.labels must be 'filtered' or 'all' and not '{args.labels}'")
    logger.info(f"Use {args.labels} labels ({args.num_labels})")

    # Load the model
    logger.info(f"Load transformer model and tokenizer ({config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME]})")
    token_cls_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
        config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME],
        num_labels=args.num_labels)

    # cross validation
    if args.routine == 'cv':
        cross_validation(args, token_cls_model)
    # simple training
    else:
        simple_training(args, token_cls_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    set_seeds(args.seed, "args")

    train_routine(args)
