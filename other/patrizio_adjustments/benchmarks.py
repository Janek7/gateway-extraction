from glob import glob
import os
from pathlib import Path
# import warnings

# from petbenchmarks.tokenclassificationsentences import SentenceLevelTokenClassificationBenchmark
from petbenchmarks.tokenclassification import TokenClassificationBenchmark
from petbenchmarks.relationsextraction import RelationsExtractionBenchmark

from petbenchmarks.Abstract import LoadJsonData
from petreader.labels import *
from petreader.TokenClassification import TokenClassification
from petreader.RelationsExtraction import RelationsExtraction

SINGLE_FILE_PREDICTIONS = 'PREDITIONS-IN-SINGLE-FILE'
MULTIPLE_FILES_PREDICTIONS = 'PREDICTIONS-IN-MULTIPLE-FILES'

# SENTENCE_LEVEL = 'SENTENCE-LEVEL-PREDICTION'
DOCUMENT_LEVEL = 'DOCUMENT-LEVEL-PREDICTION'

TOKEN_CLASSIFICATION = 'TOKEN-CLASSIFICATION'
RELATION_EXTRACTION_PREDICTIONS = 'RELATIONS-EXTRACTION-PREDICTION'
RELATION_EXTRACTION_PREDICTIONS_WITH_INDEXES = 'RELATIONS-EXTRACTION-PREDICTION-WITH-INDEXES'


class Benchmark:
    def __init__(self):
        """
        benchmark interface

        predictions are stored in a list
            each item of the list represents a prediction file

            predictions files must be of the same type!

        """
        #  set default
        self.SetComparisonParameters()

        #  store prediction file list
        self.prediction_file_list = list()

        #  store predictions
        self.predictions = list()

        #  store benchmarking tasks to perform
        self.benchmarking_level = None
        self.benchmarking_task = None
        self.output_results = None

    def SetComparisonParameters(self,
                                matching_strategy: str = WORD_BASED_STRATEGY,
                                matching_relax_window=1,
                                matching_relax_strategy='lr') -> None:

        #  which strategy adopt to compare relations
        if matching_strategy in [WORD_BASED_STRATEGY, INDEX_BASED_STRATEGY]:
            self.matching_strategy = matching_strategy
        else:
            raise ValueError('Matching strategy {} not valid. Choose one between [{}, {}]'.format(
                    matching_strategy, WORD_BASED_STRATEGY, INDEX_BASED_STRATEGY))

        #  how much relax annotations
        self.matching_relax_window = matching_relax_window
        self.matching_relax_strategy = matching_relax_strategy

    def LoadPrediction(self, path):
        if os.path.isdir(path):
            self.__LoadPredictionFolder(path)
        else:
            self.__LoadPredictionFile(path)
        self.__LoadPredictions()

    def __LoadPredictionFile(self,
                             prediction_filename: str):
        #  predictions are stored in a single file
        self.prediction_file_list.append(prediction_filename)

    def __LoadPredictionFolder(self,
                               prediction_folder):
        #  predictions are stored in multiple files.
        #  Useful for cross validation

        #  scan the folder and add all the json-like files
        for filename in glob(prediction_folder + "*"):
            if not os.path.isdir(filename):
                self.prediction_file_list.append(filename)

    def __LoadPredictions(self):
        for filename in self.prediction_file_list:
            data = LoadJsonData(filename)
            if data:
                self.predictions.append(data)
            else:
                # warnings.warn('File {} not valid'.format(filename))
                pass

    def __define_type_of_prediction(self):
        """
        try to define the type of prediction file.
        """
        #
        # if self.__Is_sentence_level():
        #
        #     self.benchmarking_level = SENTENCE_LEVEL
        #     self.benchmarking_task = TOKEN_CLASSIFICATION
        #
        # elif self.__Is_document_level():
        self.benchmarking_level = DOCUMENT_LEVEL

        if self.__Is_relation_task():
            if self.__Is_RelationExtractionWithIndexes():
                self.benchmarking_task = RELATION_EXTRACTION_PREDICTIONS_WITH_INDEXES
            else:
                self.benchmarking_task = RELATION_EXTRACTION_PREDICTIONS
        else:
            self.benchmarking_task = TOKEN_CLASSIFICATION

        # else:
        #     raise ValueError('Predictions format not valid')

    def __Is_sentence_level(self):
        if ACTIVITY in self.predictions[0] and len(self.predictions[0][ACTIVITY]) == 417:
            return True
        return False

    def __Is_relation_task(self):
        k = list(self.predictions[0].keys())[0]  # take first key (doc name)

        if USES in self.predictions[0][k].keys():
            return True
        return False

    # def __Is_document_level(self):
    #     return True
    #     #  if it is a document level, it has the key
    #     predictions = self.predictions[0]

    def __Is_RelationExtractionWithIndexes(self):
        predictions = self.predictions[0]

    def LunchBenchmark(self, pet_dataset=None, results_filename='results'):
        """
            lunch the benchmark procedure
        """
        self.__define_type_of_prediction()

        if self.benchmarking_task == TOKEN_CLASSIFICATION:
            #  token classification
            if pet_dataset:
                assert isinstance(pet_dataset, TokenClassification)
            bench = TokenClassificationBenchmark(pet_dataset=pet_dataset)
        else:
            #  relation extraction
            if pet_dataset:
                assert isinstance(pet_dataset, RelationsExtraction)
            bench = RelationsExtractionBenchmark(pet_dataset=pet_dataset)
        #  set matching strategy parameters
        bench.change_matching_strategy(self.matching_strategy)
        bench.SetMatchingFunction(relax_window=self.matching_relax_window,
                                  strategy=self.matching_relax_strategy)

        for predictions in self.predictions:
            bench.ComputeScores(predictions)

        #  performs statistics
        bench.ComputeStatistics()

        bench.ShowStatistics()

        #  save results to files
        bench.ShowStatistics(results_filename + '.txt')
        bench.SaveResults(results_filename + '.json')


class BenchmarkApproach(Benchmark):
    """
    This class implements the interface to benchmark appproaches on the PET dataset.

    """

    def __init__(self,
                 approach_name: str,
                 predictions_file_or_folder: str,
                 output_results=None,
                 relax_window=1,
                 strategy='lr',
                 pet_dataset=None,
                 ):
        """

        Args:
            approach_name:  (str) the name of the approach benchmarked
            predictions_file_or_folder: (str) path to predictions.
                                        it can be either a file or a folder containing results files
            output_results: (str) the path to the results file to store
            relax_window: (int) the number of words to consider when relaxing comparison
            strategy: (str) the strategy adopted to perform comparisong.
                            'lr': left-right
                            'l' : left
                            'r' : right
            pet_dataset: (TokenClassification or RelationsExtraction) loaded dataset instance for the task to benchmark

        """
        super(BenchmarkApproach, self).__init__()

        self.approach_name = approach_name

        if not output_results:
            self.output_results = str(
                Path(predictions_file_or_folder).parent.joinpath('results-' + approach_name).absolute())
        else:
            self.output_results = output_results

        self.matching_relax_window = relax_window
        self.matching_relax_strategy = strategy
        self.predictions_file_or_folder = predictions_file_or_folder

        self.LoadPrediction(self.predictions_file_or_folder)

        self.LunchBenchmark(pet_dataset, self.output_results)
