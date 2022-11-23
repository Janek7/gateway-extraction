#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys
# find recursively the project root dir
parent_dir = os.path.abspath(os.path.join(os.path.abspath(''), os.pardir))
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import argparse
import json
import os
import logging

import tensorflow as tf
import transformers
from petreader.labels import *

from Ensemble import Ensemble
from SameGatewayClassifier import SameGatewayClassifier
from labels import *
from metrics import compute_avg_metrics, print_metrics
from same_gateway_data_preparation import create_same_gateway_cls_dataset_full, create_same_gateway_cls_dataset_cv
from utils import config, set_seeds, get_seed_list, generate_args_logdir


logger = logging.getLogger('Same Gateway Classifier [Training]')

parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed_general", default=42, type=int, help="Random seed.")
parser.add_argument("--ensemble", default=True, type=bool, help="Use ensemble learning with config.json seeds.")
parser.add_argument("--seeds_ensemble", default="0-1", type=str, help="Random seed range to use for ensembles")

# routine params
parser.add_argument("--routine", default="cv", type=str, help="Simple split training 'sp', cross validation 'cv' or "
                                                              "full training without validation 'ft'.")
parser.add_argument("--folds", default=2, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")
# Architecture / data params
parser.add_argument("--context_size", default=1, type=int, help="Number of sentences around to include in text.")
parser.add_argument("--mode", default=CONCAT, type=str, help="How to include gateway information.")
parser.add_argument("--n_gram", default=1, type=int, help="Number of tokens to include for gateway in CONCAT mode.")


def train_routine(gateway_type: str, args: argparse.Namespace) -> None:

    # Load the model
    logger.info(f"Load transformer model and tokenizer ({config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME]})")
    bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])

    # cross validation
    if args.routine == 'cv':
        cross_validation(gateway_type, args, None)
    # full training wihtout validation
    elif args.routine == 'ft':
        full_training(gateway_type, args, bert_model)
    else:
        raise ValueError(f"Invalid training routine: {args.routine}")


def cross_validation(gateway_type: str, args: argparse.Namespace, bert_model) -> None:
    """
    run training in a cross validation routine -> averaged results are outputted into logdir
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param args: namespace args
    :param bert_model: bert like transformer model
    :return:
    """
    logger.info(f"Run {args.folds}-fold cross validation")

    folded_datasets = create_same_gateway_cls_dataset_cv(gateway_type, args, shuffle=True)

    os.makedirs(args.logdir, exist_ok=True)
    args_logdir_original = args.logdir

    metrics_per_fold = {'avg_val_loss': 0, 'avg_val_binary_accuracy': 0, 'avg_val_precision': 0, 'avg_val_recall': 0,
                        'val_loss': [], 'val_binary_accuracy': [], 'val_precision': [], 'val_recall': []}

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
            model = SameGatewayClassifier(args, bert_model, mode=args.mode, train_size=len(train_dataset))
            history = model.fit(
                train_dataset, epochs=args.epochs, validation_data=dev_dataset,
                callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq='batch', profile_batch=0)]
            )
            # store model
            if args.store_weights:
                model.save_weights(os.path.join(args.logdir, "weights/weights"))

        # b) fit ensemble model (train multiple seeds for current fold)
        else:
            ensemble_model = Ensemble(model_class=SameGatewayClassifier, seeds=get_seed_list(args.seeds_ensemble),
                                      args=args, bert_model=bert_model, mode=args.mode, train_size=len(train_dataset))
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


def full_training(gateway_type: str, args: argparse.Namespace, bert_model) -> None:
    """
    run training with full dataset without validation
    :param gateway_type: type of gateway to extract data for (XOR_GATEWAY or AND_GATEWAY)
    :param args: namespace args
    :param bert_model: bert like transformer model
    :return:
    """
    logger.info(f"Run full training")
    args_dir_original = args.logdir

    # create dataset
    train = create_same_gateway_cls_dataset_full(gateway_type, args, shuffle=True)

    if not args.ensemble:
        # train
        model = SameGatewayClassifier(args, bert_model, train_size=len(train))
        history = model.fit(
            train, epochs=args.epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq="batch", profile_batch=0)]
        )

        # store model
        if args.store_weights:
            model.save_weights(os.path.join(args.logdir, "weights/weights"))

    else:
        # train
        args_dir_original = args.logdir
        ensemble_model = Ensemble(model_class=SameGatewayClassifier, seeds=get_seed_list(args.seeds_ensemble),
                                  args=args, bert_model=bert_model, mode=args.mode, train_size=len(train))
        history = ensemble_model.fit(args, train_dataset=train, save_single_models=args.store_weights)

    # store metrics
    with open(os.path.join(args_dir_original, "metrics.json"), 'w') as file:
        json.dump(history.history, file, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.logdir = generate_args_logdir(args, script_name="SameGatewayClassifier")

    # this seed is used by default (only overwritten in case of ensemble)
    set_seeds(args.seed_general, "args - used for dataset split/shuffling")

    train_routine(XOR_GATEWAY, args)
