import argparse
import json
import logging
import os
from typing import List, Tuple, Dict

import tensorflow as tf
import numpy as np

from Ensemble import Ensemble
from metrics import f1_normal
from utils import get_seed_list

logger = logging.getLogger('Training')

# A) HELPER METHODS & DATA FOR TRAINING AND CROSS VALIDATION


# dictionary of metrics for early stopping and cross validation for each model
MONITOR_METRICS = 'monitor_metric'
METRICS_PER_FOLD = 'metrics_per_fold'

_model_metrics = {
    "GatewayTokenClassifier": {
        MONITOR_METRICS: "val_xor_precision",
        METRICS_PER_FOLD: ['avg_val_loss', 'avg_val_xor_precision', 'avg_val_xor_recall', 'avg_val_xor_f1',
                           'avg_val_xor_f1_m', 'avg_val_and_recall', 'avg_val_and_precision', 'avg_val_and_f1',
                           'avg_val_and_f1_m', 'avg_val_overall_accuracy', 'val_loss', 'val_xor_precision',
                           'val_xor_recall', 'val_xor_f1', 'val_xor_f1_m', 'val_and_recall', 'val_and_precision',
                           'val_and_f1', 'val_and_f1_m', 'val_overall_accuracy']
    },
    "SameGatewayClassifier": {
        MONITOR_METRICS: "val_binary_accuracy",
        METRICS_PER_FOLD: ['avg_val_loss', 'avg_val_binary_accuracy', 'avg_val_precision', 'avg_val_recall', 'val_loss',
                           'val_binary_accuracy', 'val_precision', 'val_recall']
    }
}


def get_empty_metrics_per_fold(model_class: type(tf.keras.Model)) -> Dict:
    """
    creates empty dictionary for metrics to monitor for each fold during cross validation of the model
    :return: dictionary
    """
    return {m: 0 if m.startswith("avg") else [] for m in _model_metrics[model_class.__name__][METRICS_PER_FOLD]}


def get_monitor(model_class: type(tf.keras.Model)) -> str:
    """
    return monitor metrics for early stopping for the given model
    :param model_class: model
    :return: metric name
    """
    return _model_metrics[model_class.__name__][MONITOR_METRICS]


def compute_avg_metrics(metrics_per_fold):
    # compute f1 score manually again
    for i in range(len(metrics_per_fold['val_loss'])):
        if 'val_xor_f1_m' in metrics_per_fold:
            metrics_per_fold['val_xor_f1_m'].append(f1_normal(metrics_per_fold['val_xor_precision'][i],
                                                              metrics_per_fold['val_xor_recall'][i]))
        if 'val_and_f1_m' in metrics_per_fold:
            metrics_per_fold['val_and_f1_m'].append(f1_normal(metrics_per_fold['val_and_precision'][i],
                                                              metrics_per_fold['val_and_recall'][i]))

    # average metrics over folds
    for metric, value in metrics_per_fold.items():
        if not metric.startswith("avg_") and not metric.startswith("seeds-last_epoch-"):
            metrics_per_fold[f"avg_{metric}"] = round(np.mean(value), 4)


def print_metrics(metrics_per_fold):
    print(' Score per fold '.center(100, '-'))
    for i in range(len(metrics_per_fold['val_loss'])):
        if i > 0: print('-' * 100)
        metric_str = ' - '.join([f"{metric}: {round(value[i], 4)}" for metric, value in metrics_per_fold.items() if
                                 not metric.startswith("avg_") and not metric.startswith("seeds-last_epoch-")])
        print(f"> Fold {i + 1} - {metric_str}")
    print()
    print(' Average scores '.center(100, '-'))
    print(' - '.join([f"{metric}: {round(value, 4)}" for metric, value in metrics_per_fold.items() if
                      metric.startswith("avg_")]))


# TRAINING FUNCTIONS


def cross_validation(args: argparse.Namespace, model_class: type(tf.keras.Model),
                     folded_datasets: List[Tuple[tf.data.Dataset, tf.data.Dataset]], bert_model=None) -> None:
    """
    run training of a given model in a cross validation routine
    averaged results are outputted into logdir
    :param args: namespace args
    :param model_class: model class to cross validate
    :param folded_datasets: list of train/dev datasets of multiple folds to use for cross validation
    :param bert_model: bert like transformer model as core of models
    :return:
    """
    logger.info(f"Run {args.folds}-fold cross validation")

    os.makedirs(args.logdir, exist_ok=True)
    args_logdir_original = args.logdir

    metrics_per_fold = get_empty_metrics_per_fold(model_class)

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
            model = model_class(args, bert_model, mode=args.mode, train_size=len(train_dataset))
            history = model.fit(
                train_dataset, epochs=args.epochs, validation_data=dev_dataset,
                callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq='batch', profile_batch=0),
                           tf.keras.callbacks.EarlyStopping(monitor=get_monitor(model_class),
                                                            min_delta=1e-4, patience=2, mode="max",
                                                            restore_best_weights=True)]
            )
            # store model
            if args.store_weights:
                model.save_weights(os.path.join(args.logdir, "weights/weights"))

        # b) fit ensemble model (train multiple seeds for current fold)
        else:
            ensemble_model = Ensemble(
                # ensemble params
                model_class=model_class, seeds=get_seed_list(args.seeds_ensemble), es_monitor=get_monitor(model_class),
                # single model params
                args=args, bert_model=bert_model, train_size=len(train_dataset)
            )
            history = ensemble_model.fit(args, train_dataset=train_dataset, dev_dataset=dev_dataset, fold=i)

        # record fold results (record only validation results; drop training metrics)
        for metric, values in history.history.items():
            # record validation value of last epoch for each metric
            if metric.startswith("val_"):
                # average seed values of last epoch to receive fold value
                metrics_per_fold[metric].append(round(np.mean(history.history[f"seeds-last_epoch-{metric}"]), 4))
            elif metric.startswith("seeds-last_epoch-val_"):
                # record seed values of last epoch to show variance
                metrics_per_fold[f"{metric}-{i}"] = values

    logger.info("Finished CV, average, print and save results")
    compute_avg_metrics(metrics_per_fold)
    print_metrics(metrics_per_fold)
    print(args)

    # save metrics
    with open(os.path.join(args_logdir_original, "cv_metrics.json"), 'w') as file:
        json.dump(metrics_per_fold, file, indent=4)


def full_training(args: argparse.Namespace, model_class: type(tf.keras.Model), dataset: tf.data.Dataset,
                  bert_model=None) -> None:
    """
    run training with full dataset without validation
    :param args: namespace args
    :param model_class: model class to cross validate
    :param dataset: train dataset
    :param bert_model: bert like transformer model as core of models
    :return:
    """
    logger.info(f"Run full training")
    args_dir_original = args.logdir

    if not args.ensemble:
        # train
        model = model_class(args, bert_model, train_size=len(dataset))
        history = model.fit(
            dataset, epochs=args.epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, update_freq="batch", profile_batch=0),
                       tf.keras.callbacks.EarlyStopping(monitor=get_monitor(model_class),
                                                        min_delta=1e-4, patience=2, mode="max",
                                                        restore_best_weights=True)]
        )

        # store model
        if args.store_weights:
            model.save_weights(os.path.join(args.logdir, "weights/weights"))

    else:
        # train
        args_dir_original = args.logdir
        ensemble_model = Ensemble(
            # ensemble params
            model_class=model_class, seeds=get_seed_list(args.seeds_ensemble), es_monitor=get_monitor(model_class),
            # single model params
            args=args, bert_model=bert_model, train_size=len(dataset)
        )
        history = ensemble_model.fit(args, train_dataset=dataset, save_single_models=args.store_weights)

    # store metrics
    with open(os.path.join(args_dir_original, "metrics.json"), 'w') as file:
        json.dump(history.history, file, indent=4)