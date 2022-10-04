from petbenchmarks.benchmarks import BenchmarkApproach


# 1) GENERATE IN RIGHT FORMAT
from petbenchmarks.tokenclassification import TokenClassificationBenchmark
from petbenchmarks.relationsextraction import RelationsExtractionBenchmark


# 1a) for process elements (GATEWAYS)
# as _activities, just hand over a list of activities (without positional information -> neither sentence, nor token)
# when 3 are true and 5 are given -> 3 TP, 2 FP
# save dict in the end as file with .json end
#  SRL results
tcb = TokenClassificationBenchmark()
# tc = tcb.PETdataset
pet_process_elements = tcb.GetEmptyPredictionsDict()

pet_process_elements[doc_name][ACTIVITY].append(activity_)

# 1b) for flow relations
# list of flow relations; each is represented by dictionary that contains source and target
reb = RelationsExtractionBenchmark()
pet_relations = reb.GetEmptyPredictionsDict()
pet_relations[doc_name][USES].append({SOURCE_ENTITY: activity_,
														  TARGET_ENTITY: ad_})

# 2) RUN EVALUATION
# use for both, element extraction and relation extraction -> recognizes on json file content

print('elements')
process_elements_filename = '/results_frameworks/raw/Kruz_process_elements.json'
BenchmarkApproach(tested_approach_name='Kruz-elements',
                  predictions_file_or_folder=process_elements_filename)