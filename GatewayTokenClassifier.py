#!/usr/bin/env python3

import argparse
import json
import os
import datetime
import re
import logging

import numpy as np
import tensorflow as tf
import transformers

from token_data_preparation import create_token_classification_dataset, create_token_classification_dataset_cv
from metrics import *

logger = logging.getLogger('Gateway Token Classifier')

parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# routine params
parser.add_argument("--routine", default="cv", type=str, help="Simple training or cross validation.")
parser.add_argument("--dev_share", default=.2, type=int, help="Share of dev dataset in simple training routine.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
# Architecture params
parser.add_argument("--extra_head", default=False, type=bool, help="Include extra cls head.")
parser.add_argument("--labels", default="filtered", type=str, help="Label set to use.")
parser.add_argument("--other_labels_weight", default=0.1, type=int, help="Sample weight for non gateway tokens.")
parser.add_argument("--zhuggingface_model_name", default="distilbert-base-uncased", type=str, help="Model checkpoint")


class GatewayTokenClassifier(tf.keras.Model):

    def __init__(self, args: argparse.Namespace, model, train_dataset: tf.data.Dataset,
                 extra_head: bool = False) -> None:

        # A) OPTIMIZER
        optimizer, lr_schedule = transformers.create_optimizer(
            init_lr=2e-5,
            num_train_steps=(len(train_dataset) // args.batch_size) * args.epochs,
            weight_decay_rate=0.01,
            num_warmup_steps=10,
        )

        # B) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[64], dtype=tf.int32),  # before: None
            "attention_mask": tf.keras.layers.Input(shape=[64], dtype=tf.int32)
        }
        bert_output = model(inputs).logits  # includes one dense layer with linear activation function
        if extra_head:
            predictions = tf.keras.layers.Dense(args.num_labels, activation=tf.nn.softmax)(bert_output)
        else:
            predictions = bert_output
        super().__init__(inputs=inputs, outputs=predictions)

        # C) COMPILE
        self.compile(optimizer=optimizer,
                     # loss=custom_loss,
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     # general accuracy of all labels (except 0 class for padding tokens)
                     weighted_metrics=[tf.metrics.SparseCategoricalAccuracy(name="overall_accuracy")],
                     # metrics for classes of interest
                     metrics=[xor_precision, xor_recall, xor_f1, and_recall, and_precision, and_f1])
        # token_cls_model.summary()
        self.summary()


def simple_training(args: argparse.Namespace, token_cls_model, tokenizer) -> None:
    """
    run a training based on a simple train / test split
    :param args: namespace args
    :param token_cls_model: token classification model
    :param tokenizer: tokenizer
    :return:
    """
    logger.info(f"Run simple training (num_labels={args.num_labels}; other_labels_weight={args.other_labels_weight}; "
                f"dev_share={args.dev_share})")
    train, dev = create_token_classification_dataset(tokenizer, other_labels_weight=args.other_labels_weight,
                                                     label_set=args.labels, dev_share=args.dev_share,
                                                     batch_size=args.batch_size)

    # Create the model and train it
    model = GatewayTokenClassifier(args, token_cls_model, train, args.extra_head)

    history = model.fit(
        train, epochs=args.epochs, validation_data=dev,
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                   # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100,
                   #                                  verbose=0, mode="max", baseline=None, restore_best_weights=True)
                   ]
    )
    with open(os.path.join(args.logdir, "metrics.json"), 'w') as file:
        json.dump(history.history, file, indent=4)


def cross_validation(args: argparse.Namespace, token_cls_model, tokenizer) -> None:
    """
    run training in a cross validation routine -> averaged results are outputted into logdir
    :param args: namespace args
    :param token_cls_model: token classification model
    :param tokenizer: tokenizer
    :return:
    """
    logger.info(f"Run {args.folds}-fold cross validation (num_labels={args.num_labels}; "
                f"other_labels_weight={args.other_labels_weight})")
    folded_datasets = create_token_classification_dataset_cv(tokenizer, other_labels_weight=args.other_labels_weight,
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
                metrics_per_fold[f"avg_{metric}"] = np.mean(value)

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
        logger.info(f"Start training of fold {i}")
        # train
        args.logdir = f"{args_logdir_original}\\{i + 1}"
        os.makedirs(args.logdir, exist_ok=True)
        model = GatewayTokenClassifier(args, token_cls_model, train_dataset)
        history = model.fit(
            train_dataset, epochs=args.epochs, validation_data=dev_dataset,
            callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)]
        )

        # record fold results
        for metric, epoch_values in history.history.items():
            # record value of last epoch for each metric
            if metric.startswith("val_"):
                metric = metric[4:]
                metrics_per_fold[metric].append(epoch_values[-1])
        # models.append(model)

        update_avg_metrics(metrics_per_fold)

    logger.info("Finished CV, print and save results")
    print_metrics(metrics_per_fold)

    # save metrics
    with open(os.path.join(args_logdir_original, "cv_metrics.json"), 'w') as file:
        json.dump(metrics_per_fold, file, indent=4)


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.random.set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

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
    logger.info(f"Load transformer model and tokenizer ({args.zhuggingface_model_name})")
    token_cls_model = transformers.TFAutoModelForTokenClassification.from_pretrained(args.zhuggingface_model_name,
                                                                                     num_labels=args.num_labels)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.zhuggingface_model_name)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    # cross validation
    if args.routine == 'cv':
        cross_validation(args, token_cls_model, tokenizer)
    # simple training
    else:
        simple_training(args, token_cls_model, tokenizer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
