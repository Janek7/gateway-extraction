#!/usr/bin/env python3

# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import logging
from abc import ABC, abstractmethod
import random
from typing import Tuple, List, Dict
import itertools
import argparse
import re
from copy import deepcopy

import tensorflow as tf
import transformers
import numpy as np

from Ensemble import Ensemble
from training import cross_validation, full_training
from PetReader import pet_reader
from labels import *
from relation_approaches.activity_relation_data_preparation import get_activity_relations
from relation_approaches.activity_relation_dataset_preparation import label_dict, \
    create_activity_relation_cls_dataset_cv, create_activity_relation_cls_dataset_full, MAX_LENGTH
from utils import GatewayExtractionException, config, generate_args_logdir, set_seeds

logger = logging.getLogger('Relation Classifier')
logger_ensemble = logging.getLogger('Relation Classifier Ensemble')

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
parser.add_argument("--folds", default=5, type=int, help="Number of folds in cross validation routine.")
parser.add_argument("--store_weights", default=False, type=bool, help="Flag if best weights should be stored.")
parser.add_argument("--test_docs", default=True, type=bool, help="Flag if predefined docs should be used as test set")
parser.add_argument("--test_share", default=0.1, type=float, help="Share of test set (only applied if not test_docs")
# Data params
parser.add_argument("--down_sample_ef", default=False, type=bool, help="Flag if eventually following samples should be"
                                                                       "down sampled to comparable number")
# Architecture params
parser.add_argument("--architecture", default=ARCHITECTURE_CUSTOM, type=str, help="Architecture variants")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")
parser.add_argument("--hidden_layer", default=32, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--warmup", default=0, type=int, help="Number of warmup steps.")
# cnn params
parser.add_argument("--cnn_blocks", default=1, type=int, help="Number of filters in CNN")
parser.add_argument("--filter_start_size", default=32, type=int,
                    help="Start (minimal) number of filters in first cnn block")
parser.add_argument("--filter_increase", default=2, type=int, help="Rate how much the number of filters should grow in "
                                                                   "each new block")
parser.add_argument("--kernel_size", default=3, type=int, help="Kernel size in CNN")
parser.add_argument("--pool_size", default=2, type=int, help="Max pooling size")
# rnn params
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="Type of RNN cell (LSTM or GRU)")
parser.add_argument("--rnn_units", default=32, type=int, help="Number of units in RNNs")
parser.add_argument("--rnn_backwards", default=False, type=bool, help="Flag if backwards should be processed as well.")


# A) RelationClassifier classes


class RelationClassifier:
    """
    abstract base class for RelationClassifiers
    """

    def __init__(self):
        logger.info(f"Initialize a {self.__class__.__name__}")

    @abstractmethod
    def predict_activity_pair(self, doc_name: str, activity_1: Tuple, activity_2: Tuple) -> str:
        """
        prediction method to classify an activity pair
        activity format: tuple of (sentence idx, word idx, token_list, 'ACTIVITY')
        :param doc_name: document in which a1 and a2 occur
        :param activity_1: activity 1
        :param activity_2: activity 2
        :return: prediction as str label (see label set above)
        """
        pass


class RandomBaselineRelationClassifier(RelationClassifier):
    """
    Dummy classifier baseline that predicts always a random label
    """

    def predict_activity_pair(self, doc_name, activity_1, activity_2) -> str:
        return random.choice(list(label_dict.keys()))


class DFBaselineRelationClassifier(RelationClassifier):
    """
    Dummy classifier baseline that predicts always a DF relation
    """

    def predict_activity_pair(self, doc_name, activity_1, activity_2) -> str:
        return DIRECTLY_FOLLOWING


class GoldstandardRelationClassifier(RelationClassifier):
    """
    Classifier that returns always gold standard relation
    """

    def __init__(self):
        super().__init__()
        self.relation_data = get_activity_relations(return_type=dict)

    def predict_activity_pair(self, doc_name: str, activity_1: Tuple, activity_2: Tuple) -> str:
        """
        predict relation of a given pair of activities in a document
        :param doc_name: document
        :param activity_1:
        :param activity_2:
        :return: relation label (see head of script)
        """
        target_relation = list(filter(lambda r: r[DOC_NAME] == doc_name and
                                                ((r[ACTIVITY_1] == activity_1 and r[ACTIVITY_2] == activity_2) or
                                                 (r[ACTIVITY_1] == activity_2 and r[ACTIVITY_2] == activity_1)),
                                      self.relation_data))
        if len(target_relation) > 1:
            logger.warning(f"{doc_name}: multiple relations of {activity_1} and {activity_2} found")
            return target_relation[0][RELATION_TYPE]
        if target_relation:
            return target_relation[0][RELATION_TYPE]
        else:
            raise GatewayExtractionException(f"{doc_name}: no relation of {activity_1} and {activity_2} found")


# A2) Neural classes


class NeuralRelationClassifier(tf.keras.Model):  # , RelationClassifier):
    """
    classification model to classify relation of two activities
    class is abstract because hidden and output layers must be defined in abstract method for different architectures
    """

    def __init__(self, args: argparse.Namespace = None, bert_model=None, train_size: int = None,
                 weights_path: str = None) -> None:
        """
        creates a NeuralRelationClassifier
        :param args: args Namespace
        :param bert_model: bert like transformer model
        :param train_size: train dataset size
        :param weights_path: path of stored weights. If set, load from there
        """
        logger.info("Create and initialize a NeuralRelationClassifier")
        if not args:
            logger.warning("No parsed args passed to NeuralRelationClassifier, use standard values")
            args = parser.parse_args([] if "__file__" not in globals() else None)
        self.args = args
        self.weights_path = weights_path

        # A) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[MAX_LENGTH], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[MAX_LENGTH], dtype=tf.int32)
        }

        if not bert_model:
            bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
        # includes one dense layer with linear activation function
        bert_output = bert_model({"input_ids": inputs["input_ids"],
                                  "attention_mask": inputs["attention_mask"]}).last_hidden_state

        predictions = self.create_hidden_and_output_layers(bert_output=bert_output)

        # tf.keras.Model.__init__(self, inputs, predictions)
        # RelationClassifier.__init__(self)
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
                         loss=tf.keras.losses.CategoricalCrossentropy(),
                         metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                                  tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

        # self.summary()

        # if model path is passed, restore weights
        if self.weights_path:
            logger.info(f"Restored weights from {weights_path}")
            self.load_weights(weights_path).expect_partial()

    @abstractmethod
    def create_hidden_and_output_layers(self, bert_output) -> tf.keras.layers.Dense:
        """
        creates sequence of hidden layers and output -> extracted to method for easy extensions with different
        architectures
        :param bert_output: BERT output of input sequence
        :return: a dense classification layer
        """
        pass

    @staticmethod
    def create_output_layer(hidden_layer) -> tf.keras.layers.Layer:
        """
        create classification head layer
        """
        return tf.keras.layers.Dense(len(label_dict), activation=tf.nn.softmax)(hidden_layer)

    def create_cnn_blocks(self, input_layer) -> tf.keras.layers.Layer:
        """
        create sequence of stacked convolutional blocks (dims are shrinking gradually because of max pooling, number of
        filters increases gradually). Every block is a sequence of [Conv -> BatchNorm -> Relu -> Max Pooling]
        :param input_layer: input to the whole block
        :return: last max pooling layer that combines all layers before
        """
        output = input_layer
        filters = int(self.args.filter_start_size / self.args.filter_increase)
        for i in range(1, self.args.cnn_blocks + 1):
            filters *= self.args.filter_increase
            cnn = tf.keras.layers.Conv1D(filters, self.args.kernel_size, 1, "same")(output)
            batch_norm = tf.keras.layers.BatchNormalization()(cnn)
            relu = tf.keras.layers.ReLU()(batch_norm)
            output = tf.keras.layers.MaxPool1D(pool_size=self.args.pool_size)(relu)
        return output

    def predict_activity_pair(self, doc_name: str, activity_1: Tuple, activity_2: Tuple):
        print("")
        raise NotImplementedError


class CustomNeuralRelationClassifier(NeuralRelationClassifier):
    """
    architecture: BERT -> dropout -> hidden -> dropout -> cls head
    """

    def create_hidden_and_output_layers(self, bert_output) -> tf.keras.layers.Dense:
        """
        creates sequence of hidden layers and output -> extracted to method for easy extensions with different
        architectures
        :param bert_output: BERT output of input sequence
        :return: a dense classification layer
        """
        cls_token = bert_output[:, 0]  # extract cls token for every sample
        dropout1 = tf.keras.layers.Dropout(self.args.dropout)(cls_token)
        hidden = tf.keras.layers.Dense(self.args.hidden_layer, activation=tf.nn.relu)(dropout1)
        dropout2 = tf.keras.layers.Dropout(self.args.dropout)(hidden)
        predictions = self.create_output_layer(dropout2)
        return predictions


class CNNRelationClassifier(NeuralRelationClassifier):
    """
    architecture: BERT -> dropout -> CNN -> max pooling -> dropout -> hidden -> dropout -> cls head
    """

    def create_hidden_and_output_layers(self, bert_output) -> tf.keras.layers.Dense:
        """
        creates sequence of hidden layers and output -> extracted to method for easy extensions with different
        architectures
        :param bert_output: BERT output of input sequence
        :return: a dense classification layer
        """
        dropout1 = tf.keras.layers.Dropout(self.args.dropout)(bert_output)
        cnn = self.create_cnn_blocks(dropout1)
        cnn_flattened = tf.keras.layers.Flatten()(cnn)
        dropout2 = tf.keras.layers.Dropout(self.args.dropout)(cnn_flattened)
        hidden = tf.keras.layers.Dense(self.args.hidden_layer, activation=tf.nn.relu)(dropout2)
        dropout3 = tf.keras.layers.Dropout(self.args.dropout)(hidden)
        predictions = self.create_output_layer(dropout3)
        return predictions


class BRCNNRelationClassifier(NeuralRelationClassifier):
    """
    architecture: BERT -> 2(forward/backward) x [RNN -> CNN-> max pooling] -> concat -> cls head
    """

    def create_hidden_and_output_layers(self, bert_output) -> tf.keras.layers.Dense:
        """
        creates sequence of hidden layers and output -> extracted to method for easy extensions with different
        architectures
        :param bert_output: BERT output of input sequence
        :return: a dense classification layer
        """
        rnn_cell_type = tf.keras.layers.LSTM if self.args.rnn_cell == 'LSTM' else tf.keras.layers.GRU
        dropout1 = tf.keras.layers.Dropout(self.args.dropout)(bert_output)

        forward = rnn_cell_type(self.args.rnn_units, return_sequences=True)(dropout1)
        forward_cnn = self.create_cnn_blocks(forward)
        forward_cnn_flattened = tf.keras.layers.Flatten()(forward_cnn)
        hidden = forward_cnn_flattened
        if self.args.rnn_backwards:
            backward = rnn_cell_type(self.args.rnn_units, return_sequences=True, go_backwards=True)(dropout1)
            backward_cnn = self.create_cnn_blocks(backward)
            backward_cnn_flattened = tf.keras.layers.Flatten()(backward_cnn)
            concatenated = tf.keras.layers.Concatenate()([forward_cnn_flattened, backward_cnn_flattened])
            hidden = concatenated
        dropout2 = tf.keras.layers.Dropout(self.args.dropout)(hidden)
        predictions = self.create_output_layer(dropout2)
        return predictions


# A3) ENSEMBLE

architecture_dict = {
    ARCHITECTURE_CUSTOM: CustomNeuralRelationClassifier,
    ARCHITECTURE_CNN: CNNRelationClassifier,
    ARCHITECTURE_BRCNN: BRCNNRelationClassifier
}


class NeuralRelationClassifierEnsemble(Ensemble):
    """
    Ensemble (seeds) of activity relation classification model
    - During training normal Ensemble class is used
    - Used only for inference -> extends normal Ensemble by classification methods
    """

    def __init__(self, ensemble_path: str = None, seeds: List = None, es_monitor: str = 'val_loss',
                 seed_limit: int = None, **kwargs) -> None:
        """
        see super class for param description
        override for fixing model class
        :param log_folder: log_folder where to store results
        """
        self.predictions = {}
        # in case of reload of ensemble args are not passed -> create args, extract used architecture from path and set
        if ensemble_path and "args" not in kwargs:
            logger.info("Use standard values of args when reloading ensemble")
            args = parser.parse_args([] if "__file__" not in globals() else None)
            mode_pattern = re.compile("[-,]a=([a-zA-Z_]+)")
            args.architecture = mode_pattern.search(ensemble_path).group(1)
            logger.info(f"Reload model with architecture {args.architecture}")
            kwargs["args"] = args

        if kwargs["args"].architecture not in [ARCHITECTURE_CUSTOM, ARCHITECTURE_CNN, ARCHITECTURE_BRCNN]:
            raise ValueError(f"{kwargs['args'].architecture} is not a valid architecture")

        super().__init__(architecture_dict[kwargs["args"].architecture], seeds, ensemble_path, es_monitor, seed_limit,
                         **kwargs)

    def predict_test_set(self, test: tf.data.Dataset) -> np.ndarray:
        """
        predict labels of given test set
        :param test: tensorflow data set
        :return: numpy array with numeric labels
        """
        predictions = [m.predict(test) for m in self.models]
        predictions_averaged = np.mean(predictions, axis=0)
        predictions_argmax = np.argmax(predictions_averaged, axis=-1)
        return predictions_argmax


# B) TRAINING

def train_routine(args: argparse.Namespace) -> None:
    """
    run SameGatewayClassifier training based on passed args
    :param args: namespace args
    :return:
    """

    # Load the model
    # logger.info(f"Load transformer model ({config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME]})")
    # bert_model = transformers.TFAutoModel.from_pretrained(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])

    # different architectures are implemented as subclasses -> choose already here
    logger.info(f"Train model with architecture {args.architecture}")
    model_class = architecture_dict[args.architecture]

    # cross validation
    if args.routine == 'cv':
        folded_datasets = create_activity_relation_cls_dataset_cv(args)
        cross_validation(args, model_class, folded_datasets)

    # full training without validation
    elif args.routine == 'ft':
        train, test, test_relations = create_activity_relation_cls_dataset_full(args)
        full_training(args, model_class, train)

    else:
        raise ValueError(f"Invalid training routine: {args.routine}")


# C) APPLICATION OF RelationClassifier


def classify_documents(relation_classifier: RelationClassifier, doc_names: List[str] = None) -> Dict:
    """
    evaluate a RelationClassifier by applying it to given list of doc_names (if empty, evaluate all)
    :param relation_classifier: RelationClassifier instance
    :param doc_names: documents to evaluate
    :return: dictionary with {doc_name: list of activity relations)
    """
    relations = {}
    if not doc_names:
        doc_names = pet_reader.document_names
    logger.info(f"Create activity relation predictions for {len(doc_names)} documents using a "
                f"{relation_classifier.__class__.__name__}")

    # Create predictions for every activity pair in every document in given list
    for i, doc_name in enumerate(doc_names):

        if i % 5 == 0:
            logger.info(f"Processed {i} documents")

        doc_relations = []
        doc_activities = pet_reader.get_activities_in_relation_approach_format(doc_name)
        for a1, a2 in itertools.combinations(doc_activities, 2):
            doc_relations.append((a1, a2, relation_classifier.predict_activity_pair(doc_name, a1, a2)))

        relations[doc_name] = doc_relations

    return relations


def create_relation_benchmark_format(predictions: np.ndarray, original_relations: List) -> Dict[str, List]:
    """
    maps predictions to the input relations (copy input list and replace labels) and save in dictionary with format used
    in RelationClassificationBenchmark -> {doc name: List[(activity1, activity2, relation type), ...]}
    :param predictions: numpy array with numerical labels
    :param original_relations: list of input relations (dict format)
    :return: predicted relations organized in dictionary per document
    """
    # create relations
    relation_predictions = deepcopy(original_relations)
    for relation, prediction in zip(relation_predictions, predictions):
        relation[RELATION_TYPE] = [textual for textual, numeric in label_dict.items() if numeric == prediction][0]

    # organize in dict {doc_name: doc_relations}
    involved_doc_names = list(set([r[DOC_NAME] for r in relation_predictions]))
    relation_predictions_dict = {doc_name: [(r[ACTIVITY_1], r[ACTIVITY_2], r[RELATION_TYPE])
                                            for r in relation_predictions if r[DOC_NAME] == doc_name]
                                 for doc_name in involved_doc_names}
    return relation_predictions_dict


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.logdir = generate_args_logdir(args, script_name="RelationClassifier")

    # this seed is used by default (only overwritten in case of ensemble)
    set_seeds(args.seed_general, "args - used for dataset split/shuffling")

    # goldstandard_classifier = GoldstandardRelationClassifier()
    # relations = classify_documents(goldstandard_classifier, ["doc-9.5"])
    #
    # for doc_name, doc_relations in relations.items():
    #     print(doc_name.center(100, '-'))
    #     for r in doc_relations:
    #         print(r)

    train_routine(args)

    # train, test, test_relations = create_activity_relation_cls_dataset_full(args)

    # ensemble = NeuralRelationClassifierEnsemble(log_folder="TEST", seeds=[1,2], args=args)
