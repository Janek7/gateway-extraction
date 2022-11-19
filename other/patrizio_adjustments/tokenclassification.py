import codecs
from copy import deepcopy
import numpy as np
import os
import pickle
# import warnings

from petreader.TokenClassification import TokenClassification
from petbenchmarks.Abstract import AbstractBenchmark
from petreader.labels import *


class TokenClassificationBenchmark(AbstractBenchmark):
    """
        Class to benchmarking approaches on the PET dataset

        This class benchmark relations extraction task

        It adopts two strategy:
            - Word based:
                    the comparison between goldstandard and predictions is done by comparing words
                    provided in the predicted results file.
                    It is agnostic to sentence and word position.

            - Index base;
                    the comparison is done by comparing strictly comparing the index of the source and target sentences
                    start and ending index of word-span
    """

    def __init__(self, pet_dataset=None):
        super(TokenClassificationBenchmark, self).__init__()
        #
        # # store strategy
        # self.change_matching_strategy(matching_strategy)

        #  load PET data
        if pet_dataset:
            assert isinstance(pet_dataset, TokenClassification)
            self.PETdataset = pet_dataset
        else:
            self.PETdataset = TokenClassification()
    #
    # def __init__(self):
    #     super(TokenClassificationBenchmark, self).__init__()
    #     #
    #     # # store strategy
    #     # self.change_matching_strategy(matching_strategy)
    #
    #     #  load PET data
    #     self.PETdataset = TokenClassification()

        ####### temp Janek code #######
        goldstandard_object_path = "data/other/token_goldstandard.pkl"
        # if already exists, load pickled object
        if os.path.isfile(goldstandard_object_path):
            with open(os.path.abspath(goldstandard_object_path), 'rb') as file:
                self.goldstandard = pickle.load(file)
        else:
            self.goldstandard = self.GetGoldStandard()
            with open(os.path.abspath(goldstandard_object_path), 'wb') as file:
                pickle.dump(self.goldstandard, file, protocol=pickle.HIGHEST_PROTOCOL)
        ####### temp Janek code #######

        # self.goldstandard = self.GetGoldStandard()
        #  load goldstandard data
        # self.goldstandard = self.GetGoldStandard(tc)
        # self.empty_dict = self._get_empty_dict(tc)

    def GetEmptyPredictionsDict(self) -> dict:
        #     return self.empty_dict
        #
        # def _get_empty_dict(self, tc) -> dict:
        return {k: {ACTIVITY:                list(),
                    ACTIVITY_DATA:           list(),
                    ACTOR:                   list(),
                    FURTHER_SPECIFICATION:   list(),
                    XOR_GATEWAY:             list(),
                    AND_GATEWAY:             list(),
                    CONDITION_SPECIFICATION: list()} for k in self.PETdataset.GetDocumentNames()}

    # def GetGoldStandard(self, tc) -> dict:
    def GetGoldStandard(self) -> dict:
        """
            dict(doc_name: {preocess element: [sent1, sent2, ...

        :return:
        """

        def _get_items_(items_collection):
            flatten_item = list()

            for items_in_sentence in items_collection:
                for item_in_sentence in items_in_sentence:
                    flatten_item.append(item_in_sentence)

            return flatten_item

        #######################
        golds = dict()

        for doc_name in self.PETdataset.GetDocumentNames():
            golds[doc_name] = {ACTIVITY:                _get_items_(self.PETdataset.GetActivities(doc_name)),
                               ACTIVITY_DATA:           _get_items_(self.PETdataset.GetActivityDatas(doc_name)),
                               ACTOR:                   _get_items_(self.PETdataset.GetActors(doc_name)),
                               FURTHER_SPECIFICATION:   _get_items_(self.PETdataset.GetFurtherSpecifications(doc_name)),
                               XOR_GATEWAY:             _get_items_(self.PETdataset.GetXORGateways(doc_name)),
                               AND_GATEWAY:             _get_items_(self.PETdataset.GetANDGateways(doc_name)),
                               CONDITION_SPECIFICATION: _get_items_(
                                   self.PETdataset.GetConditionSpecifications(doc_name))}

        return golds

    def ComputeScores(self,
                      predictions):
        # warnings.warn('to be tested')
        #  for each document of the predictions
        for doc_name in self.goldstandard.keys():
            print('benchmarking {}'.format(doc_name))
            document_scores = dict()
            # for each relation label
            for element_label in PROCESS_ELEMENT_LABELS:
                #  collect tp, fp, tn, fn
                try:
                    if predictions[doc_name][element_label]:
                        # print(doc_name, element_label, predictions[doc_name][element_label],
                        #       len(predictions[doc_name][element_label]), type(predictions[doc_name][element_label]))
                        scores = self.ComputeClassEntityScores(self.goldstandard[doc_name][element_label],
                                                               predictions[doc_name][element_label])

                    else:  # in case there are no predictions
                        scores = self.ComputeClassEntityScores(self.goldstandard[doc_name][element_label], None)

                except KeyError:
                    scores = self.ComputeClassEntityScores(self.goldstandard[doc_name][element_label], None)

                scores[SUPPORT] = sum([len(sent) for sent in self.PETdataset.get_element(
                        document_name=doc_name,
                        process_element=element_label)])
                #  add class label to scores
                document_scores[element_label] = scores

            #  add results_frameworks to global results_frameworks
            self.bench_scores[doc_name] = document_scores

    def ComputeStatistics(self):
        """Compute stastistics.
        """

        #  secure copy, remove point reference
        global_scores = deepcopy(self.bench_scores)

        #  compute global statistics
        supports_statistics = self.Supports(global_scores)
        average_statistics = self.__AverageStatistics(global_scores)
        micro_statistics = self.__MicroStatistics(global_scores)
        macro_statistics = self.__MacroStatistics(global_scores)

        self.bench_scores.update(supports_statistics)
        self.bench_scores.update(micro_statistics)
        self.bench_scores.update(macro_statistics)
        self.bench_scores.update(average_statistics)

    @staticmethod
    def Supports(results: dict) -> dict:
        #  return a dict with the sum of supports
        #  it returns a dict to conform with the other statistics methods
        supports = [results[doc_name][processele_type][SUPPORT] for doc_name in results for processele_type in
                    results[doc_name]]
        return {SUPPORT: int(np.sum(supports))}
    #
    @staticmethod
    def Support(results: dict, process_element=ACTIVITY) -> int:
        #  return a dict with the sum of supports
        #  it returns a dict to conform with the other statistics methods
        supports = [results[doc_name][process_element][SUPPORT] for doc_name in results]
        # return {SUPPORT: int(np.sum(supports))}
        return int(np.sum(supports))

    @staticmethod
    def __MacroPerClassStatistics(results: dict) -> dict:
        #  results_frameworks a dict of the global results_frameworks

        #  The macro-averaged F1 score (or macro F1 score) is computed by taking
        #  the arithmetic mean (aka unweighted mean) of all the per-class F1 scores.

        statistics = dict()

        #  pre-class statistics
        for processele_type in PROCESS_ELEMENT_LABELS:
            precisions = [results[doc_name][processele_type][PRECISION] for doc_name in results]
            recalls = [results[doc_name][processele_type][RECALL] for doc_name in results]
            f1s = [results[doc_name][processele_type][F1SCORE] for doc_name in results]
            supports = [results[doc_name][processele_type][SUPPORT] for doc_name in results]

            macro_precision = float(np.mean(precisions))
            macro_recall = float(np.mean(recalls))
            macro_f1 = float(np.mean(f1s))
            support = int(np.sum(supports))

            statistics[processele_type] = {PRECISION: macro_precision,
                                           RECALL:    macro_recall,
                                           F1SCORE:   macro_f1,
                                           SUPPORT:   support}

        return statistics

    def __MacroStatistics(self,
                          results: dict) -> dict:
        #  results_frameworks a dict of the global results_frameworks

        #  The macro-averaged F1 score (or macro F1 score) is computed by taking
        #  the arithmetic mean (aka unweighted mean) of all the per-class F1 scores.

        statistics = dict()

        #  pre-class statistics
        perclass = self.__MacroPerClassStatistics(results)

        #  overall
        macro_precision = float(np.mean([perclass[k][PRECISION] for k in PROCESS_ELEMENT_LABELS]))
        macro_recall = float(np.mean([perclass[k][RECALL] for k in PROCESS_ELEMENT_LABELS]))
        macro_f1 = float(np.mean([perclass[k][F1SCORE] for k in PROCESS_ELEMENT_LABELS]))
        supports = self.Supports(results)

        statistics[OVERALL_STATISTICS] = {PRECISION: macro_precision,
                                          RECALL:    macro_recall,
                                          F1SCORE:   macro_f1,
                                          SUPPORT:   supports}

        statistics[PER_CLASS_STATISTICS] = perclass

        return {MACRO_STATISTICS: statistics}

    @staticmethod
    def __AveragePerClassStatistics(results: dict) -> dict:
        """Add Average statistics to results.

            Compute Average statistics:
                    - Average of class type results

        Args:
            results (dict):
                comparison results
        Returns:
            dict with statistics
        """
        # results_frameworks a dict of the global results_frameworks

        #  The weighted-averaged F1 score is calculated by taking the mean of all per-class per document
        #  F1 scores while considering each class’s support.
        statistics = dict()

        #  pre-class statistics
        for rel_type in PROCESS_ELEMENT_LABELS:
            precisions = [results[doc_name][rel_type][PRECISION] for doc_name in results]
            recalls = [results[doc_name][rel_type][RECALL] for doc_name in results]
            f1s = [results[doc_name][rel_type][F1SCORE] for doc_name in results]
            supports = [results[doc_name][rel_type][SUPPORT] for doc_name in results]

            average_precision = float(np.average(precisions, weights=supports))
            average_recall = float(np.average(recalls, weights=supports))
            average_f1 = float(np.average(f1s, weights=supports))
            support = int(np.sum(supports))

            statistics[rel_type] = {PRECISION: average_precision,
                                    RECALL:    average_recall,
                                    F1SCORE:   average_f1,
                                    SUPPORT:   support}

        return statistics

    def __AverageStatistics(self,
                            results: dict) -> dict:
        """Add Average statistics to results.

            Compute Average statistics:
                    - Average of class type results
                    - overall statistics

        Args:
            results (dict):
                comparison results
        Returns:
            dict with statistics
        """
        # results_frameworks a dict of the global results_frameworks

        #  The weighted-averaged F1 score is calculated by taking the mean of all per-class
        #  F1 scores while considering each class’s support.
        statistics = dict()
        perclass = self.__AveragePerClassStatistics(results)

        #  overall
        precisions = [results[doc_name][rel_type][PRECISION] for doc_name in results for rel_type in results[doc_name]]
        recalls = [results[doc_name][rel_type][RECALL] for doc_name in results for rel_type in results[doc_name]]
        f1s = [results[doc_name][rel_type][F1SCORE] for doc_name in results for rel_type in results[doc_name]]
        supports = [results[doc_name][rel_type][SUPPORT] for doc_name in results for rel_type in results[doc_name]]

        average_precision = float(np.average(precisions, weights=supports))
        average_recall = float(np.average(recalls, weights=supports))
        average_f1 = float(np.average(f1s, weights=supports))
        supports = self.Supports(results)

        statistics[OVERALL_STATISTICS] = {PRECISION: average_precision,
                                          RECALL:    average_recall,
                                          F1SCORE:   average_f1,
                                          SUPPORT:   supports}
        statistics[PER_CLASS_STATISTICS] = perclass

        return {AVERAGE_STATISTICS: statistics}

    def __MicroPerClassStatistics(self, results: dict) -> dict:
        #  results_frameworks a dict of the global results_frameworks

        # Micro averaging computes a global average F1 score by counting the sums of the True Positives (TP),
        # False Negatives (FN), and False Positives (FP).
        statistics = dict()

        #  pre-class statistics
        for process_element in PROCESS_ELEMENT_LABELS:
            tps = [results[doc_name][process_element][TRUE_POSITIVE] for doc_name in results]
            TP = int(np.sum(tps))
            fps = [results[doc_name][process_element][FALSE_POSITIVE] for doc_name in results]
            FP = int(np.sum(fps))
            fns = [results[doc_name][process_element][FALSE_NEGATIVE] for doc_name in results]
            FN = int(np.sum(fns))

            micro_precision = self.Precision(tp=TP, fp=FP)
            micro_recall = self.Recall(tp=TP, fn=FN)
            micro_f1 = self.F1(TP, FP, FN)
            support = self.PETdataset.Statistics()[process_element] #Support(results=results, process_element=process_element)  #tp=TP, fn=FN)

            statistics[process_element] = {PRECISION: micro_precision,
                                    RECALL:    micro_recall,
                                    F1SCORE:   micro_f1,
                                    SUPPORT:   support}

        return statistics

    def __MicroStatistics(self, results: dict) -> dict:
        #  results_frameworks a dict of the global results_frameworks

        # Micro averaging computes a global average F1 score by counting the sums of the True Positives (TP),
        # False Negatives (FN), and False Positives (FP).
        statistics = dict()

        perclass = self.__MicroPerClassStatistics(results)

        #  overall
        tps = [results[doc_name][rel_type][TRUE_POSITIVE] for doc_name in results for rel_type in results[doc_name]]
        TP = int(np.sum(tps))
        fps = [results[doc_name][rel_type][FALSE_POSITIVE] for doc_name in results for rel_type in results[doc_name]]
        FP = int(np.sum(fps))
        fns = [results[doc_name][rel_type][FALSE_NEGATIVE] for doc_name in results for rel_type in results[doc_name]]
        FN = int(np.sum(fns))

        micro_precision = self.Precision(tp=TP, fp=FP)
        micro_recall = self.Recall(tp=TP, fn=FN)
        micro_f1 = self.F1(TP, FP, FN)
        supports = self.Supports(results)

        statistics[OVERALL_STATISTICS] = {PRECISION: micro_precision,
                                          RECALL:    micro_recall,
                                          F1SCORE:   micro_f1,
                                          SUPPORT:   supports}

        statistics[PER_CLASS_STATISTICS] = perclass

        return {MICRO_STATISTICS: statistics}

    def __ShowDocumentStatistics(self,
                                 results: dict,
                                 foout=None) -> None:
        for class_tag in PROCESS_ELEMENT_LABELS:
            print('{class_:<25} precision: {pr:1.2f} | recall: {rec:1.2f} | f1: {f1:1.2f} | supports: {sup:n}'.format(
                    class_=class_tag,
                    pr=self.round(results[class_tag][PRECISION]),
                    rec=self.round(results[class_tag][RECALL]),
                    f1=self.round(results[class_tag][F1SCORE]),
                    sup=results[class_tag][SUPPORT]),
                    file=foout)

    def ShowStatistics(self,
                       filename=None):
        if filename:
            foout = codecs.open(filename, 'w', 'utf-8')
        else:
            foout = None

        results = self.bench_scores
        #  Show document's statistics
        print('Documents Statistiscs',
              file=foout)
        #  collect doc_name list and exclude statistics items
        doc_list = [doc_name for doc_name in results.keys()
                    if doc_name not in [AVERAGE_STATISTICS, MACRO_STATISTICS, MICRO_STATISTICS, SUPPORT]]

        for doc_name in sorted(doc_list):
            print('',
                  file=foout)
            print('Document: {}'.format(doc_name),
                  file=foout)
            self.__ShowDocumentStatistics(results[doc_name], foout)
            self.print_sep(foout)

        print(file=foout)
        print('Per-class Statistics',
              file=foout)
        self.print_sep(foout)
        for statistics_type in [MICRO_STATISTICS, AVERAGE_STATISTICS, MACRO_STATISTICS]:
            print('Statistics: {}'.format(statistics_type),
                  file=foout)
            self.print_sep()
            self.__ShowDocumentStatistics(results[statistics_type][PER_CLASS_STATISTICS], foout)
            print(file=foout)

        print('Overall statistics',
              file=foout)
        self.print_sep(foout)
        for statistics_type in [MICRO_STATISTICS, AVERAGE_STATISTICS, MACRO_STATISTICS]:
            print('{class_:<25} precision: {pr:1.2f} | recall: {rec:1.2f} | f1: {f1:1.2f}'.format(
                    class_=statistics_type,
                    pr=self.round(results[statistics_type][OVERALL_STATISTICS][PRECISION]),
                    rec=self.round(results[statistics_type][OVERALL_STATISTICS][RECALL]),
                    f1=self.round(results[statistics_type][OVERALL_STATISTICS][F1SCORE])),
                    file=foout)

        print('{:<25} {}'.format(SUPPORT, results[SUPPORT]),
              file=foout)

        if foout:
            foout.close()


if __name__ == '__main__':
    bench = TokenClassificationBenchmark()
    golds = bench.GetGoldStandard()
