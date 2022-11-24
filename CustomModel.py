from typing import Dict


class CustomModel:
    """
    super class for tensorflow models that can be used in an ensemble and trained with cross validation
    contains static attributes that provide information for training models
    """

    _monitor_metric = None
    _metrics_per_fold = None

    @classmethod
    def get_empty_metrics_per_fold(cls) -> Dict:
        """
        creates empty dictionary for metrics to monitor for each fold during cross validation of the model
        :return: dictionary
        """
        return {m: 0 if m.startswith("avg") else [] for m in cls._metrics_per_fold}

    @classmethod
    def get_monitor(cls):
        return cls._monitor_metric
