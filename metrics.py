import numpy as np
from keras import backend as K
import tensorflow as tf

from labels import *


# HELPER METHODS FOR CROSS VALIDATION

def compute_avg_metrics(metrics_per_fold):
    # compute f1 score manually again
    for i in range(len(metrics_per_fold['val_xor_precision'])):
        if 'val_xor_f1_m' in metrics_per_fold:
            metrics_per_fold['val_xor_f1_m'].append(f1_normal(metrics_per_fold['val_xor_precision'][i],
                                                              metrics_per_fold['val_xor_recall'][i]))
        if 'val_and_f1_m' in metrics_per_fold:
            metrics_per_fold['val_and_f1_m'].append(f1_normal(metrics_per_fold['val_and_precision'][i],
                                                              metrics_per_fold['val_and_recall'][i]))

    # average metrics over folds
    for metric, value in metrics_per_fold.items():
        if not metric.startswith("avg_") and not metric.startswith("seed-results-"):
            metrics_per_fold[f"avg_{metric}"] = round(np.mean(value), 4)


def print_metrics(metrics_per_fold):
    print(' Score per fold '.center(100, '-'))
    for i in range(len(metrics_per_fold['val_loss'])):
        if i > 0: print('-' * 100)
        metric_str = ' - '.join([f"{metric}: {round(value[i], 4)}" for metric, value in metrics_per_fold.items() if
                                 not metric.startswith("avg_") and not metric.startswith("seed-results-")])
        print(f"> Fold {i + 1} - {metric_str}")
    print()
    print(' Average scores '.center(100, '-'))
    print(' - '.join([f"{metric}: {round(value, 4)}" for metric, value in metrics_per_fold.items() if
                      metric.startswith("avg_")]))


# CUSTOM METRICS FOR TOKEN CLS


def filter_y_for_target_label(y_true, y_pred, target_label):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true_filtered = tf.where(tf.equal(y_true, target_label), 1, tf.zeros_like(y_true))
    y_pred_filtered = tf.where(tf.equal(y_pred, target_label), 1, tf.zeros_like(y_pred))
    return y_true_filtered, y_pred_filtered


def TPs(y_true, y_pred):
    """ assume y_true and y_pred with 0 and 1 values for binary classification """
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def f1(precision, recall):
    return 2 * (tf.math.multiply(precision, recall) / (precision + recall + K.epsilon()))


def f1_normal(precision, recall):
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def xor_precision(y_true, y_pred):
    if len(tf.shape(y_pred)) == 3:
        y_pred = tf.math.argmax(y_pred, axis=2)
    y_true_filtered, y_pred_filtered = filter_y_for_target_label(y_true, y_pred, TC_LABEL_XOR)
    true_positives = TPs(y_true_filtered, y_pred_filtered)
    predicted_positives = K.sum(K.round(K.clip(y_pred_filtered, 0, 1)))
    if predicted_positives == 0:
        return tf.constant(0.0, dtype=tf.float32)
    else:
        return tf.cast(true_positives / predicted_positives, tf.float32)


def xor_recall(y_true, y_pred):
    if len(tf.shape(y_pred)) == 3:
        y_pred = tf.math.argmax(y_pred, axis=2)
    y_true_filtered, y_pred_filtered = filter_y_for_target_label(y_true, y_pred, TC_LABEL_XOR)
    true_positives = TPs(y_true_filtered, y_pred_filtered)
    real_positives = K.sum(K.round(K.clip(y_true_filtered, 0, 1)))
    if real_positives == 0:
        return tf.constant(0.0, dtype=tf.float32)
    else:
        return tf.cast(true_positives / real_positives, tf.float32)


def xor_f1(y_true, y_pred):
    precision = xor_precision(y_true, y_pred)
    recall = xor_recall(y_true, y_pred)
    return tf.cast(f1(precision, recall), tf.float32)


def and_precision(y_true, y_pred):
    if len(tf.shape(y_pred)) == 3:
        y_pred = tf.math.argmax(y_pred, axis=2)
    y_true_filtered, y_pred_filtered = filter_y_for_target_label(y_true, y_pred, TC_LABEL_AND)
    true_positives = TPs(y_true_filtered, y_pred_filtered)
    predicted_positives = K.sum(K.round(K.clip(y_pred_filtered, 0, 1)))
    if predicted_positives == 0:
        return tf.constant(0.0, dtype=tf.float32)
    else:
        return tf.cast(true_positives / predicted_positives, tf.float32)


def and_recall(y_true, y_pred):
    if len(tf.shape(y_pred)) == 3:
        y_pred = tf.math.argmax(y_pred, axis=2)
    y_true_filtered, y_pred_filtered = filter_y_for_target_label(y_true, y_pred, TC_LABEL_AND)
    true_positives = TPs(y_true_filtered, y_pred_filtered)
    real_positives = K.sum(K.round(K.clip(y_true_filtered, 0, 1)))
    if real_positives == 0:
        return tf.constant(0.0, dtype=tf.float32)
    else:
        return tf.cast(true_positives / real_positives, tf.float32)


def and_f1(y_true, y_pred):
    precision = and_precision(y_true, y_pred)
    recall = and_recall(y_true, y_pred)
    return tf.cast(f1(precision, recall), tf.float32)


if __name__ == '__main__':
    y_true1 = tf.cast(tf.constant([[1, 1, 0],
                                   [0, 2, 1]]), tf.int32)
    y_pred1 = tf.cast(tf.constant([[1, 1, 0],
                                   [2, 2, 2]]), tf.int32)

    print(xor_precision(y_true1, y_pred1))
    print(xor_recall(y_true1, y_pred1))
    print(xor_f1(y_true1, y_pred1))
