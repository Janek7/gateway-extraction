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

from GatewayTokenClassifier import GatewayTokenClassifier, GatewayTokenClassifierEnsemble
from token_data_preparation import create_token_classification_dataset, create_token_classification_dataset_cv, \
    create_full_training_dataset
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
parser.add_argument("--routine", default="ft", type=str, help="Simple split training 'sp', cross validation 'cv' or "
                                                              "full training without validation 'ft'.")
parser.add_argument("--dev_share", default=0.1, type=float, help="Share of dev dataset in simple training routine.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--sampling_strategy", default=NORMAL, type=str, help="How to sample samples.")
parser.add_argument("--store_weights", default=True, type=bool, help="Flag if best weights should be stored.")
# Architecture params
parser.add_argument("--ensemble", default=True, type=bool, help="Use ensemble learning with config.json seeds.")
parser.add_argument("--labels", default=ALL, type=str, help="Label set to use.")
parser.add_argument("--other_labels_weight", default=0.1, type=float, help="Sample weight for non gateway tokens.")


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
    print(args)

    # Load the model
    logger.info(f"Load transformer model and tokenizer ({config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME]})")
    token_cls_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
        config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME],
        num_labels=args.num_labels)

    # cross validation
    if args.routine == 'cv':
        cross_validation(args, token_cls_model)
    # simple split training
    elif args.routine == 'sp':
        simple_split_training(args, token_cls_model)
    # full training wihtout validation
    elif args.routine == 'ft':
        full_training(args, token_cls_model)
    else:
        raise ValueError(f"Invalid training routine: {args.routine}")


def simple_split_training(args: argparse.Namespace, token_cls_model) -> None:
    """
    run a training based on a simple train / test split
    :param args: namespace args
    :param token_cls_model: token classification model
    :return:
    """
    logger.info(f"Run simple training (num_labels={args.num_labels}; other_labels_weight={args.other_labels_weight}; "
                f"dev_share={args.dev_share})")
    train, dev = create_token_classification_dataset(args.sampling_strategy,
                                                     other_labels_weight=args.other_labels_weight,
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
        model.save_weights(os.path.join(args.logdir, "weights/weights"))

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
        folded_datasets = create_token_classification_dataset_cv(args.sampling_strategy,
                                                                 other_labels_weight=args.other_labels_weight,
                                                                 label_set=args.labels, kfolds=args.folds,
                                                                 batch_size=args.batch_size)
    else:
        seed_dataset_lists = []
        for seed in config[ENSEMBLE_SEEDS]:
            set_seeds(seed, "GatewayTokenClassifierEnsemble - dataset creation")
            seed_dataset_lists.append(
                create_token_classification_dataset_cv(args.sampling_strategy,
                                                       other_labels_weight=args.other_labels_weight,
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
            fold_i_seed_train_datasets = [seed_folded_datasets[i][0] for seed_folded_datasets in seed_dataset_lists]
            fold_i_seed_dev_datasets = [seed_folded_datasets[i][1] for seed_folded_datasets in seed_dataset_lists]
            ensemble_model = GatewayTokenClassifierEnsemble(args, token_cls_model,
                                                            train_size=len(fold_i_seed_train_datasets[0]))
            # history = ensemble_model.fit(args, fold_i_seed_datasets)
            history = ensemble_model.fit(args, fold_i_seed_train_datasets, fold_i_seed_dev_datasets)

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


def full_training(args: argparse.Namespace, token_cls_model) -> None:
    logger.info(f"Run full training (num_labels={args.num_labels}; other_labels_weight={args.other_labels_weight})")

    if not args.ensemble:
        # create dataset
        train = create_full_training_dataset(args.sampling_strategy, other_labels_weight=args.other_labels_weight,
                                             label_set=args.labels, batch_size=args.batch_size)

        # train
        model = GatewayTokenClassifier(args, token_cls_model, len(train))
        history = model.fit(
            train, epochs=args.epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq="batch", profile_batch=0),
                       tf.keras.callbacks.EarlyStopping(monitor='val_overall_accuracy', min_delta=1e-4, patience=1,
                                                        verbose=0, mode="max", restore_best_weights=True)]
        )

        # store model
        if args.store_weights:
            model.save_weights(os.path.join(args.logdir, "weights/weights"))

        # store metrics
        with open(os.path.join(args.logdir, "metrics.json"), 'w') as file:
            json.dump(history.history, file, indent=4)

    else:
        # create datasets
        train_datasets = []
        for seed in config[ENSEMBLE_SEEDS]:
            set_seeds(seed, "GatewayTokenClassifierEnsemble - dataset creation")
            train_datasets.append(create_full_training_dataset(args.sampling_strategy,
                                                               other_labels_weight=args.other_labels_weight,
                                                               label_set=args.labels, batch_size=args.batch_size))
        # train
        args_dir_original = args.logdir
        ensemble_model = GatewayTokenClassifierEnsemble(args, token_cls_model,
                                                        train_size=len(train_datasets[0]))
        history = ensemble_model.fit(args, train_datasets, save_single_models=args.store_weights)

        # store metrics
        with open(os.path.join(args_dir_original, "metrics.json"), 'w') as file:
            json.dump(history.history, file, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    set_seeds(args.seed, "args")

    # train_routine(args)

    ensemble = GatewayTokenClassifierEnsemble(ensemble_path="C:\\Users\\janek\\Development\\Git\\master-thesis\\data\\logs\\GatewayTokenClassifier_train.py-2022-11-10_070547-bs=8,ds=0.1,e=True,e=1,f=2,l=all,olw=0.1,r=ft,s=42,sw=True")
