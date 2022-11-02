#!/usr/bin/env python3

import argparse
import json
import os
import datetime
import re
import logging
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from token_data_preparation import create_token_classification_dataset, create_token_classification_dataset_cv
from metrics import *
from utils import config, set_seeds

logger = logging.getLogger('Gateway Token Classifier')

parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# routine params
parser.add_argument("--routine", default="cv", type=str, help="Simple training or cross validation.")
parser.add_argument("--dev_share", default=0.1, type=float, help="Share of dev dataset in simple training routine.")
parser.add_argument("--folds", default=5, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")
# Architecture params
parser.add_argument("--extra_head", default=False, type=bool, help="Include extra cls head.")
parser.add_argument("--labels", default="filtered", type=str, help="Label set to use.")
parser.add_argument("--other_labels_weight", default=0.1, type=float, help="Sample weight for non gateway tokens.")


class GatewayTokenClassifier(tf.keras.Model):

    def __init__(self, args: argparse.Namespace = None, bert_model=None, train_dataset: tf.data.Dataset = None,
                 extra_head: bool = False, weights_path: str = None) -> None:
        """
        creates a GatewayTokenClassifier
        :param args: args Namespace
        :param bert_model: bert like transformer for token classification model
        :param train_dataset: train dataset
        :param extra_head: flag if to include an extra classification head on top of the loaded model
        :param weights_path: path of stored weights. If set, load from there
        """
        self.extra_head = extra_head
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
        bert_output = bert_model(inputs).logits  # includes one dense layer with linear activation function
        if extra_head:
            predictions = tf.keras.layers.Dense(num_labels, activation=tf.nn.softmax)(bert_output)
        else:
            predictions = bert_output
        super().__init__(inputs=inputs, outputs=predictions)

        # B) COMPILE (only needed when training is intended)
        if args and train_dataset:
            optimizer, lr_schedule = transformers.create_optimizer(
                init_lr=2e-5,
                num_train_steps=(len(train_dataset) // args.batch_size) * args.epochs,
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

    def predict(self, tokens: transformers.BatchEncoding, word_ids: List[List[int]]) \
            -> List[List[int]]:
        """
        create predictions for given data; output is one (numerical) label for each input token
        :param tokens: tokens as BatchEncoding
        :param word_ids: original word ids of tokens as 2-dim list
        :return:
        """
        predictions = super().predict({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})

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
    model = GatewayTokenClassifier(args, token_cls_model, train, args.extra_head)

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


def cross_validation(args: argparse.Namespace, token_cls_model, store_model: bool = False) -> None:
    """
    run training in a cross validation routine -> averaged results are outputted into logdir
    :param args: namespace args
    :param token_cls_model: token classification model
    :param store_model: flag if model weights should be stored in logdir
    :return:
    """
    logger.info(f"Run {args.folds}-fold cross validation (num_labels={args.num_labels}; "
                f"other_labels_weight={args.other_labels_weight})")
    folded_datasets = create_token_classification_dataset_cv(other_labels_weight=args.other_labels_weight,
                                                             label_set=args.labels, kfolds=args.folds,
                                                             batch_size=args.batch_size)
    args_logdir_original = args.logdir

    # models = []  # not used because of memory limitations
    metrics_per_fold = {'avg_loss': 0, 'avg_xor_precision': 0, 'avg_xor_recall': 0, 'avg_xor_f1': 0,
                        'avg_and_recall': 0, 'avg_and_precision': 0, 'avg_and_f1': 0, 'avg_overall_accuracy': 0,
                        'loss': [], 'xor_precision': [], 'xor_recall': [], 'xor_f1': [], 'and_recall': [],
                        'and_precision': [], 'and_f1': [], 'overall_accuracy': []}

    def update_avg_metrics(metrics_per_fold):
        for metric, value in metrics_per_fold.items():
            if not metric.startswith("avg_"):
                metrics_per_fold[f"avg_{metric}"] = round(np.mean(value), 4)

    def print_metrics(metrics_per_fold):
        print(' Score per fold '.center(50, '-'))
        for i in range(len(metrics_per_fold['loss'])):
            if i > 0: print('-' * 50)
            metric_str = ' - '.join([f"{metric}: {round(value[i], 4)}" for metric, value in metrics_per_fold.items() if
                                     not metric.startswith("avg_")])
            print(f"> Fold {i + 1} - {metric_str}")
        print()
        print(' Average scores '.center(50, '-'))
        print(' - '.join([f"{metric}: {round(value, 4)}" for metric, value in metrics_per_fold.items() if
                          metric.startswith("avg_")]))

    # perform k-fold CV
    for i, (train_dataset, dev_dataset) in enumerate(folded_datasets):
        logger.info(f" Start training of fold {i} ".center(100, '-'))

        # train
        args.logdir = f"{args_logdir_original}/{i + 1}"
        os.makedirs(args.logdir, exist_ok=True)
        model = GatewayTokenClassifier(args, token_cls_model, train_dataset)
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

        # record fold results
        for metric, epoch_values in history.history.items():
            # record value of last epoch for each metric
            if metric.startswith("val_"):
                metric = metric[4:]
                metrics_per_fold[metric].append(round(epoch_values[-1], 4))
        # models.append(model)

        update_avg_metrics(metrics_per_fold)

    logger.info("Finished CV, print and save results")
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
    set_seeds(args.seed)

    train_routine(args, )
