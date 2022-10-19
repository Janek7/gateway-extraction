import argparse
import os
import datetime
import re

import tensorflow as tf
import tensorflow_addons as tfa
import transformers

from data_preparation import create_token_classification_dataset
from labels import *

parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# Architecture params
parser.add_argument("--huggingface_model_name", default="distilbert-base-uncased", type=str, help="Model checkpoint")


class GatewayTokenClassifier(tf.keras.Model):

    def __init__(self, args: argparse.Namespace, model, train_dataset: tf.data.Dataset) -> None:

        # A) OPTIMIZER
        optimizer, lr_schedule = transformers.create_optimizer(
            init_lr=2e-5,
            num_train_steps=(len(train_dataset) // args.batch_size) * args.epochs,
            weight_decay_rate=0.01,
            num_warmup_steps=0,
        )

        # B) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        }
        predictions = model(inputs)
        super().__init__(inputs=inputs, outputs=predictions)

        # C) COMPILE
        self.compile(optimizer=optimizer,
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy"),
                              #                                  tf.metrics.Precision(name="precision"),
                              #                                  tf.metrics.Recall(name="recall"),
                              #                                  tfa.metrics.F1Score(4, name="f1")
                              ])
        self.summary()


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)

    # Create logdir name
    args.logdir = os.path.join("data/logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the model
    token_cls_model = transformers.TFAutoModelForTokenClassification.from_pretrained(args.huggingface_model_name,
                                                                                     num_labels=len(GTC_LABELS))
    # Load the data
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.huggingface_model_name)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    train, dev = create_token_classification_dataset(tokenizer)

    # Create the model and train it
    model = GatewayTokenClassifier(args, token_cls_model, train)
    print(args)
    model.fit(
        train, batch_size=args.batch_size, epochs=args.epochs, validation_data=dev,
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                   # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100,
                   #                                  verbose=0, mode="max", baseline=None, restore_best_weights=True)
                   ]
    )
    print(args)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    # os.makedirs(args.logdir, exist_ok=True)
    # with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
    #     # : Predict the tags on the test set.
    #     print("create test set predictions")
    #     predictions = model.predict(test)
    #
    #     label_strings = facebook.test.label_mapping.get_vocabulary()
    #     for sentence in predictions:
    #         print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
