#!/usr/bin/env python3

import argparse
import json
import os
import datetime
import re
import logging

import tensorflow as tf
import transformers

from GatewayTokenClassifier import GatewayTokenClassifier, GatewayTokenClassifierEnsemble
from token_data_preparation import create_token_cls_dataset_cv, create_token_cls_dataset_full
from metrics import *
from utils import config, set_seeds, get_seed_list

logger = logging.getLogger('Gateway Token Classifier')
logger_ensemble = logging.getLogger('Gateway Token Classifier Ensemble')

parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed_general", default=42, type=int, help="Random seed.")
parser.add_argument("--seeds_ensemble", default="0-1", type=str, help="Random seed range to use for ensembles")
# routine params
parser.add_argument("--routine", default="cv", type=str, help="Simple split training 'sp', cross validation 'cv' or "
                                                              "full training without validation 'ft'.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")
# Architecture / data params
parser.add_argument("--ensemble", default=True, type=bool, help="Use ensemble learning with config.json seeds.")
parser.add_argument("--labels", default=ALL, type=str, help="Label set to use.")
parser.add_argument("--other_labels_weight", default=0.1, type=float, help="Sample weight for non gateway tokens.")
parser.add_argument("--sampling_strategy", default=NORMAL, type=str, help="How to sample samples.")
parser.add_argument("--use_synonyms", default=False, type=str, help="Include synonym samples.")
parser.add_argument("--activity_usage", default=NOT, type=str, help="How to include activity data.")


def train_routine(args: argparse.Namespace) -> None:
    # Create logdir name
    args.logdir = os.path.join("../data/logs", "{}-{}-{}".format(
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
        config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME], num_labels=args.num_labels)

    # cross validation
    if args.routine == 'cv':
        cross_validation(args, token_cls_model)
    # full training wihtout validation
    elif args.routine == 'ft':
        full_training(args, token_cls_model)
    else:
        raise ValueError(f"Invalid training routine: {args.routine}")


def cross_validation(args: argparse.Namespace, token_cls_model) -> None:
    """
    run training in a cross validation routine -> averaged results are outputted into logdir
    :param args: namespace args
    :param token_cls_model: bert like transformer token classification model
    :return:
    """
    logger.info(f"Run {args.folds}-fold cross validation")

    folded_datasets = create_token_cls_dataset_cv(args)

    os.makedirs(args.logdir, exist_ok=True)
    args_logdir_original = args.logdir

    # models = []  # not used because of memory limitations
    metrics_per_fold = {'avg_val_loss': 0, 'avg_val_xor_precision': 0, 'avg_val_xor_recall': 0, 'avg_val_xor_f1': 0, 'avg_val_xor_f1_m': 0,
                        'avg_val_and_recall': 0, 'avg_val_and_precision': 0, 'avg_val_and_f1': 0, 'avg_val_and_f1_m': 0, 'avg_val_overall_accuracy': 0,
                        'val_loss': [], 'val_xor_precision': [], 'val_xor_recall': [], 'val_xor_f1': [], 'val_xor_f1_m': [],
                        'val_and_recall': [], 'val_and_precision': [], 'val_and_f1': [], 'val_and_f1_m': [], 'val_overall_accuracy': []}

    # perform k-fold CV

    for i in range(args.folds):
        logger.info(f" Start training of fold {i} ".center(100, '-'))

        # create fold folder
        if not args.ensemble:
            args.logdir = f"{args_logdir_original}/{i + 1}"
            os.makedirs(args.logdir, exist_ok=True)

        train_dataset, dev_dataset = folded_datasets[i][0], folded_datasets[i][1]
        # a) fit normal models
        if not args.ensemble:
            model = GatewayTokenClassifier(args, token_cls_model, len(train_dataset))
            history = model.fit(
                train_dataset, epochs=args.epochs, validation_data=dev_dataset,
                callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq='batch', profile_batch=0),
                           # tf.keras.callbacks.EarlyStopping(monitor='val_overall_accuracy', min_delta=1e-4, patience=1,
                           #                                  verbose=0, mode="max", restore_best_weights=True)
                           ]
            )
            # store model
            if args.store_weights:
                model.save_weights(os.path.join(args.logdir, "weights/weights"))

        # b) fit ensemble model (train multiple seeds for current fold)
        else:
            ensemble_model = GatewayTokenClassifierEnsemble(args, token_cls_model, train_size=len(train_dataset),
                                                            seeds=get_seed_list(args.seeds_ensemble))
            history = ensemble_model.fit(args, train_dataset=train_dataset, dev_dataset=dev_dataset, fold=i)

        # record fold results (record only validation results; drop training metrics)
        for metric, values in history.history.items():
            # record validation value of last epoch for each metric
            if metric.startswith("val_"):
                # values = values of metric in each epoch (in case of ensemble, values is already averaged over seeds)
                metrics_per_fold[metric].append(round(values[-1], 4))
            elif metric.startswith("seed-results-val_"):
                # values = list of results of metric for every seed (last epoch)
                metrics_per_fold[f"{metric}-{i}"] = values

    logger.info("Finished CV, average, print and save results")
    compute_avg_metrics(metrics_per_fold)
    print_metrics(metrics_per_fold)

    # save metrics
    with open(os.path.join(args_logdir_original, "cv_metrics.json"), 'w') as file:
        json.dump(metrics_per_fold, file, indent=4)


def full_training(args: argparse.Namespace, token_cls_model) -> None:
    """
    run training with full dataset without validation
    :param args: namespace args
    :param token_cls_model: bert like transformer token classification model
    :return:
    """
    logger.info(f"Run full training (num_labels={args.num_labels}; other_labels_weight={args.other_labels_weight})")

    # create dataset
    train = create_token_cls_dataset_full(args)

    if not args.ensemble:
        # train
        model = GatewayTokenClassifier(args, token_cls_model=token_cls_model, train_size=len(train))
        history = model.fit(
            train, epochs=args.epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq="batch", profile_batch=0)
                       # tf.keras.callbacks.EarlyStopping(monitor='val_overall_accuracy', min_delta=1e-4, patience=1,
                       #                                  verbose=0, mode="max", restore_best_weights=True)
                                             ]
        )

        # store model
        if args.store_weights:
            model.save_weights(os.path.join(args.logdir, "weights/weights"))

        # store metrics
        with open(os.path.join(args.logdir, "metrics.json"), 'w') as file:
            json.dump(history.history, file, indent=4)

    else:
        # train
        args_dir_original = args.logdir
        ensemble_model = GatewayTokenClassifierEnsemble(args, token_cls_model=token_cls_model, train_size=len(train),
                                                        seeds=get_seed_list(args.seeds_ensemble))
        history = ensemble_model.fit(args, train_dataset=train, save_single_models=args.store_weights)

        # store metrics
        with open(os.path.join(args_dir_original, "metrics.json"), 'w') as file:
            json.dump(history.history, file, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # this seed is used by default (only overwritten in case of ensemble)
    set_seeds(args.seed_general, "args - used for dataset split/shuffling")

    train_routine(args)
