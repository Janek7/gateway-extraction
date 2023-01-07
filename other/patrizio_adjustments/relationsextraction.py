import codecs
from copy import deepcopy
import numpy as np
# import warnings
import os
import pickle

from petreader.RelationsExtraction import RelationsExtraction
from petbenchmarks.Abstract import AbstractBenchmark
from petreader.labels import *


class RelationsExtractionBenchmark(AbstractBenchmark):
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
        super(RelationsExtractionBenchmark, self).__init__()
        #
        # # store strategy
        # self.change_matching_strategy(matching_strategy)

        #  load PET data
        if pet_dataset:
            assert isinstance(pet_dataset, RelationsExtraction)
            self.PETdataset = pet_dataset
        else:
            self.PETdataset = RelationsExtraction()
    #
    # def __init__(self):
    #     super(RelationsExtractionBenchmark, self).__init__()
    #     #
    #     # # store strategy
    #     # self.change_matching_strategy(matching_strategy)
    #
    #     #  load PET data
    #     self.PETdataset = RelationsExtraction()

        ####### temp Janek code #######
        statistics_object_path = "data/other/relations_statistics.pkl"
        # if already exists, load pickled object
        if os.path.isfile(statistics_object_path):
            with open(os.path.abspath(statistics_object_path), 'rb') as file:
                self.statistics = pickle.load(file)
        else:
            self.statistics = self.PETdataset.Statistics()
            with open(os.path.abspath(statistics_object_path), 'wb') as file:
                pickle.dump(self.statistics, file, protocol=pickle.HIGHEST_PROTOCOL)

        goldstandard_object_path = "data/other/relations_goldstandard.pkl"
        # if already exists, load pickled object
        if os.path.isfile(goldstandard_object_path):
            with open(os.path.abspath(goldstandard_object_path), 'rb') as file:
                self.goldstandard = pickle.load(file)
        else:
            self.goldstandard = self.GetGoldStandard()
            with open(os.path.abspath(goldstandard_object_path), 'wb') as file:
                pickle.dump(self.goldstandard, file, protocol=pickle.HIGHEST_PROTOCOL)
        ####### temp Janek code #######

        # self.statistics = self.PETdataset.Statistics()

        #  load goldstandard data
        # self.goldstandard = self.GetGoldStandard()

    def GetEmptyPredictionsDict(self) -> dict:
        #  get the list of document
        document_names = self.PETdataset.GetDocumentNames()

        #  data variable that host goldstandard
        data = {k: dict() for k in document_names}

        #  for each document
        for doc_name in document_names:
            relations = {k: list() for k in RELATIONS_EXTRACTION_LABELS}
            data[doc_name] = relations
        return data

    def GetGoldStandard(self) -> dict:

        #  get the list of document
        document_names = self.PETdataset.GetDocumentNames()

        #  data variable that host goldstandard
        data = {k: dict() for k in document_names}

        #  for each document
        for doc_name in document_names:
            # get document id
            doc_id = self.PETdataset.GetDocumentNumber(doc_name)
            relations = self.PETdataset.GetRelations(doc_id)

            data[doc_name] = relations

        return data

    # def GetPredicted(self) -> dict:
    #     #  returns predicted entities of the framework under test
    #     warnings.warn('DEV MODE')
    #     # dev test gold
    #     return self.GetGoldStandard()
    #
    #     return self.predicted_results

    def ComputeScores(self,
                      predictions):

        #  for each document of the predictions
        for doc_name in predictions.keys():
            print('benchmarking {}'.format(doc_name))
            document_scores = dict()
            # for each relation label
            for relation_label in RELATIONS_EXTRACTION_LABELS:
                #  collect tp, fp, tn, fn
                scores = self.__Get_Relation_TP_FP_TN_FN(self.goldstandard[doc_name][relation_label],
                                                         predictions[doc_name][relation_label],
                                                         self.matching_strategy)
                #  compute local scores
                scores = self._ComputeClassScores(scores)
                scores[SUPPORT] = self.PETdataset.Statistics(document_name=doc_name)[relation_label]

                #  add class label to scores
                document_scores[relation_label] = scores

            #  add results_frameworks to global results_frameworks
            self.bench_scores[doc_name] = document_scores

    def __Get_Relation_TP_FP_TN_FN(self,
                                   goldstandard,
                                   predictions,
                                   strategy: str = 'word-based',
                                   ) -> dict:

        #  secure copy of arrays
        goldstandard = deepcopy(goldstandard)
        predictions = deepcopy(predictions)

        #  define counter
        tp = 0
        fp = 0
        fn = 0
        #  there are no tn at all, so it is 0
        tn = 0

        for gold_item in goldstandard:
            #  check on each possible predicted item
            for n_relation_pred, _ in enumerate(predictions):

                # print(gold_item)
                # print(predictions[n_relation_pred])

                if strategy == WORD_BASED_STRATEGY:
                    gold_source = gold_item[SOURCE_ENTITY]
                    gold_target = gold_item[TARGET_ENTITY]

                    pred_source = predictions[n_relation_pred][SOURCE_ENTITY]
                    pred_target = predictions[n_relation_pred][TARGET_ENTITY]

                    #  compare source entity
                    if self.CompareItem(gold_source,
                                        pred_source):
                        #  compare target entity
                        if self.CompareItem(gold_target,
                                            pred_target):
                            #  match found !
                            tp += 1
                            #  remove match
                            predictions.remove(predictions[n_relation_pred])
                            break

                    ##############################

                else:
                    # strategy == INDEX_BASED_STRATEGY:

                    #  first sentences indexes are checked
                    if gold_item[SOURCE_SENTENCE_ID] == predictions[n_relation_pred][SOURCE_SENTENCE_ID] and \
                       gold_item[TARGET_SENTENCE_ID] == predictions[n_relation_pred][TARGET_SENTENCE_ID]:
                        # the comparison is done adopting the word strategy, using word indexes instead of word string

                        #  create entity indexes ranges
                        gold_source = [n for n in range(
                                    gold_item[SOURCE_HEAD_TOKEN_ID],
                                    gold_item[SOURCE_HEAD_TOKEN_ID]+len(gold_item[SOURCE_ENTITY]))]

                        gold_target = [n for n in range(
                                    gold_item[TARGET_HEAD_TOKEN_ID],
                                    gold_item[TARGET_HEAD_TOKEN_ID]+len(gold_item[TARGET_ENTITY]))]

                        pred_source = [n for n in range(
                            predictions[n_relation_pred][SOURCE_HEAD_TOKEN_ID],
                            predictions[n_relation_pred][SOURCE_HEAD_TOKEN_ID] +
                            len(predictions[n_relation_pred][SOURCE_ENTITY]))]

                        pred_target = [n for n in range(
                            predictions[n_relation_pred][TARGET_HEAD_TOKEN_ID],
                            predictions[n_relation_pred][TARGET_HEAD_TOKEN_ID] +
                            len(predictions[n_relation_pred][TARGET_ENTITY]))]

                        #  compare source entity
                        if self.CompareItem(gold_source,
                                            pred_source):
                            #  compare target entity
                            if self.CompareItem(gold_target,
                                                pred_target):
                                #  match found !
                                tp += 1
                                #  remove match
                                predictions.remove(predictions[n_relation_pred])
                                break

                    ##############################
            # The else clause executes after the loop completes normally.
            # This means that the loop did not encounter a break statement.
            else:
                # no match found
                #  so, it is a false negative
                fn += 1

        fn = len(goldstandard) - tp
        fp = len(predictions)  # remaining predictions without a match in goldstandard

        return {TRUE_POSITIVE: tp,
                FALSE_POSITIVE: fp,
                TRUE_NEGATIVE: tn,
                FALSE_NEGATIVE: fn}

    def ComputeStatistics(self):
        """Compute stastistics.
        """

        #  secure copy, remove point reference
        global_scores = deepcopy(self.bench_scores)

        #  compute global statistics
        supports_statistics = self.Supports(global_scores)
        average_statistics = self.__AverageStatistics(global_scores)
        micro_statistics = self.__MicroStatistics(global_scores)
        macro_statistics = self.__MacroStatisctics(global_scores)

        self.bench_scores.update(supports_statistics)
        self.bench_scores.update(micro_statistics)
        self.bench_scores.update(macro_statistics)
        self.bench_scores.update(average_statistics)

    @staticmethod
    def Support(results: dict, process_relation=USES) -> int:
        #  return a dict with the sum of supports
        #  it returns a dict to conform with the other statistics methods
        supports = [results[doc_name][process_relation][SUPPORT] for doc_name in results]
        # return {SUPPORT: int(np.sum(supports))}
        return int(np.sum(supports))

    @staticmethod
    def Supports(results: dict) -> dict:
        #  return a dict with the sum of supports
        #  it returns a dict to conform with the other statistics methods
        supports = [results[doc_name][rel_type][SUPPORT] for doc_name in results for rel_type in results[doc_name]]
        return {SUPPORT: int(np.sum(supports))}

    @staticmethod
    def __MacroPerClassStatistics(results: dict) -> dict:
        #  results_frameworks a dict of the global results_frameworks

        #  The macro-averaged F1 score (or macro F1 score) is computed by taking
        #  the arithmetic mean (aka unweighted mean) of all the per-class F1 scores.

        statistics = dict()

        #  pre-class statistics
        for rel_type in RELATIONS_EXTRACTION_LABELS:
            precisions = [results[doc_name][rel_type][PRECISION] for doc_name in results]
            recalls = [results[doc_name][rel_type][RECALL] for doc_name in results]
            f1s = [results[doc_name][rel_type][F1SCORE] for doc_name in results]
            supports = [results[doc_name][rel_type][SUPPORT] for doc_name in results]

            macro_precision = float(np.mean(precisions))
            macro_recall = float(np.mean(recalls))
            macro_f1 = float(np.mean(f1s))
            support = int(np.sum(supports))

            statistics[rel_type] = {PRECISION: macro_precision,
                                    RECALL: macro_recall,
                                    F1SCORE: macro_f1,
                                    SUPPORT: support}

        return statistics

    def __MacroStatisctics(self,
                           results: dict) -> dict:
        #  results_frameworks a dict of the global results_frameworks

        #  The macro-averaged F1 score (or macro F1 score) is computed by taking
        #  the arithmetic mean (aka unweighted mean) of all the per-class F1 scores.

        statistics = dict()

        #  pre-class statistics
        perclass = self.__MacroPerClassStatistics(results)

        #  overall
        macro_precision = float(np.mean([perclass[k][PRECISION] for k in RELATIONS_EXTRACTION_LABELS]))
        macro_recall = float(np.mean([perclass[k][RECALL] for k in RELATIONS_EXTRACTION_LABELS]))
        macro_f1 = float(np.mean([perclass[k][F1SCORE] for k in RELATIONS_EXTRACTION_LABELS]))
        supports = self.Supports(results)

        statistics[OVERALL_STATISTICS] = {PRECISION: macro_precision,
                                          RECALL: macro_recall,
                                          F1SCORE: macro_f1,
                                          SUPPORT: supports}

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
        for rel_type in RELATIONS_EXTRACTION_LABELS:
            precisions = [results[doc_name][rel_type][PRECISION] for doc_name in results]
            recalls = [results[doc_name][rel_type][RECALL] for doc_name in results]
            f1s = [results[doc_name][rel_type][F1SCORE] for doc_name in results]
            supports = [results[doc_name][rel_type][SUPPORT] for doc_name in results]

            average_precision = float(np.average(precisions, weights=supports))
            average_recall = float(np.average(recalls, weights=supports))
            average_f1 = float(np.average(f1s, weights=supports))
            support = int(np.sum(supports))

            statistics[rel_type] = {PRECISION: average_precision,
                                    RECALL: average_recall,
                                    F1SCORE: average_f1,
                                    SUPPORT: support}

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
                                          RECALL: average_recall,
                                          F1SCORE: average_f1,
                                          SUPPORT: supports}
        statistics[PER_CLASS_STATISTICS] = perclass

        return {AVERAGE_STATISTICS: statistics}

    def __MicroPerClassStatistics(self, results: dict) -> dict:
        #  results_frameworks a dict of the global results_frameworks

        # Micro averaging computes a global average F1 score by counting the sums of the True Positives (TP),
        # False Negatives (FN), and False Positives (FP).
        statistics = dict()

        #  pre-class statistics
        for rel_type in RELATIONS_EXTRACTION_LABELS:
            tps = [results[doc_name][rel_type][TRUE_POSITIVE] for doc_name in results]
            TP = int(np.sum(tps))
            fps = [results[doc_name][rel_type][FALSE_POSITIVE] for doc_name in results]
            FP = int(np.sum(fps))
            fns = [results[doc_name][rel_type][FALSE_NEGATIVE] for doc_name in results]
            FN = int(np.sum(fns))

            micro_precision = self.Precision(tp=TP, fp=FP)
            micro_recall = self.Recall(tp=TP, fn=FN)
            micro_f1 = self.F1(TP, FP, FN)
            support = self.statistics[rel_type] #self.Support(tp=TP, fn=FN)

            statistics[rel_type] = {PRECISION: micro_precision,
                                    RECALL: micro_recall,
                                    F1SCORE: micro_f1,
                                    SUPPORT: support}

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
                                          RECALL: micro_recall,
                                          F1SCORE: micro_f1,
                                          SUPPORT: supports}

        statistics[PER_CLASS_STATISTICS] = perclass

        return {MICRO_STATISTICS: statistics}

    def __ShowDocumentStatistics(self,
                                 results: dict,
                                 foout=None) -> None:
        for class_tag in RELATIONS_EXTRACTION_LABELS:
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

    #
    # def change_matching_strategy(self,
    #                              matching_strategy: str = WORD_BASED_STRATEGY) -> None:
    #     """Change the matching strategy.
    #
    #     Args:
    #         matching_strategy (str):
    #             it can be one of 'word-base' or 'index-based'.
    #
    #     """
    #
    #     # store strategy
    #     if matching_strategy in [WORD_BASED_STRATEGY, INDEX_BASED_STRATEGY]:
    #         self.matching_strategy = matching_strategy
    #     else:
    #         raise ValueError('Matching strategy {} not valid. Choose one between [{}, {}]'. format(
    #             matching_strategy, WORD_BASED_STRATEGY, INDEX_BASED_STRATEGY))


if __name__ == '__main__':
    bench = RelationsExtractionBenchmark()
    # scores = bench.ComputeScores()
    #
    # results = bench.ComputeStatistics(scores)
    # bench.ShowStatistics(results)
    #
    # bench.SaveResults(results, 'test-results_frameworks-rel.json')
    #
    # results = AbstractBenchmark.LoadJsonResults('test-results_frameworks-rel.json')
    #
    # #  print statistics
    # with codecs.open('test out.txt', 'a', 'utf-8') as out:
    #     bench.ShowStatistics(results, out)
    #
    # #  get empty prediction dict
    # empty_pred = bench.GetEmptyPredictionsDict()
    # bench.SaveResults(empty_pred, 'RelationsExtraction-prediction-empty-dict.json')
