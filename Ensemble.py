import logging
import os
import re
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from CustomModel import CustomModel
from utils import set_seeds

logger = logging.getLogger('Ensemble')


class Ensemble:

    def __init__(self, model_class: type, seeds: List = None, ensemble_path: str = None, **kwargs) -> None:
        """
        initializes a ensemble of multiple models of a passed model class
        :param model_class: class of models from which to create an ensemble
                            must inherit from CustomModel and tf.keras.Model
        :param seeds: list of seeds for which to create models (default: config seeds)
        :param ensemble_path: path of trained ensemble with stored weights. If set, load model weights from there
        :param kwargs: param list that will be passed to constructor of single models
        """
        logger.info(f"Create and initialize a Ensemble for model {model_class.__name__}")
        assert issubclass(model_class, CustomModel)
        assert issubclass(model_class, tf.keras.Model)
        self.model_class = model_class
        if seeds:
            self.seeds = seeds
        elif not seeds and ensemble_path:
            seed_range_pattern = re.compile("se=(\d+)-(\d+)")
            result = seed_range_pattern.search(ensemble_path)
            self.seeds = list(range(int(result.group(1)), int(result.group(2))))
        logger.info(f"Use {len(self.seeds)} seeds: {self.seeds}")
        self.ensemble_path = ensemble_path
        self.models = []

        # create single models based on seeds or reload from saved weights
        if self.ensemble_path:
            logger.info(f"Restore weights from trained ensemble {self.ensemble_path}")
        for i, seed in enumerate(self.seeds):
            # set only if weights are not loaded from path
            if not self.ensemble_path:
                set_seeds(seed, "Ensemble - model initialization")
            model = self.model_class(**kwargs)
            # if path to trained ensemble is passed, restore weights
            if self.ensemble_path:
                path = os.path.join(self.ensemble_path, str(seed), "weights/weights")
                model.load_weights(path).expect_partial()
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
            logger.info(f" Fold {fold}: Fit Model {i} with seed {self.seeds[i]} ".center(50, '*'))
            set_seeds(seed, "Ensemble - model fit")

            if save_single_models:
                args.logdir = f"{args_logdir_original}/{seed}"
                os.makedirs(args.logdir, exist_ok=True)

            history = model.fit(train_dataset, epochs=args.epochs, validation_data=dev_dataset,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                            min_delta=1e-4, patience=2, mode="max",
                                                                            restore_best_weights=True)])
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
