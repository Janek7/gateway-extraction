# A) CONFIG KEYS

SEED = 'general-seed'
KEYWORDS_FILTERED_APPROACH = 'keywords-filtered-approach'
BERT_MODEL_NAME = 'bert-model-name'
LABEL_SET = 'label-set'
OTHER_LABELS_WEIGHT = 'other-labels-weight'
NUM_LABELS = 'num-labels'
SYNONYM_SAMPLES_START_NUMBER = 'synonym-samples-start-number'

# B) LABELS FOR KEYWORDSAPPROACH

# set of keywords available
LITERATURE = 'literature'
LITERATURE_FILTERED = 'literature_filtered'
GOLD = 'gold'
CUSTOM = 'own'

# output formats
PET = 'pet'
BENCHMARK = 'benchmark'

# for handling surrounding activities
ELEMENT = 'element'
SOURCE = 'source'
TARGET = 'target'

# gateway frame keys for merging sequence flows
START_SENTENCE_IDX = 'start_sentence_idx'
START_TOKEN_ID = 'start_token_idx'
START_ENTITY = 'start_entity'
END_SENTENCE_IDX = 'end_sentence_idx'
END_TOKEN_ID = 'end_token_idx'
END_ENTITY = 'end_entity'


# C) LABELS FOR TOKEN CLASSIFICATION TASK

# label sets
ALL = 'all'
FILTERED = 'filtered'

# classification labels
TC_LABEL_OUT_OF_SCOPE = 0
TC_LABEL_OTHER = 1  # in case of 'filtered' labels -> all the rest; in case of 'all' labels -> 'O'
TC_LABEL_XOR = 2
TC_LABEL_AND = 3
TC_LABEL_ACTIVITY = 4
TC_LABEL_ACTIVITY_DATA = 5
TC_LABEL_ACTOR = 6
TC_LABEL_FURTHER_SPECIFICATION = 7
TC_LABEL_CONDITION_SPECIFICATION = 8
TC_LABELS_FILTERED = [TC_LABEL_OUT_OF_SCOPE, TC_LABEL_OTHER, TC_LABEL_XOR, TC_LABEL_AND]
TC_LABELS_ALL = [TC_LABEL_OUT_OF_SCOPE, TC_LABEL_OTHER, TC_LABEL_XOR, TC_LABEL_AND, TC_LABEL_ACTIVITY,
                 TC_LABEL_ACTIVITY_DATA, TC_LABEL_ACTOR, TC_LABEL_FURTHER_SPECIFICATION,
                 TC_LABEL_CONDITION_SPECIFICATION]

# weights for token classification task
TC_WEIGHTS_BERT_TOKENS = 0
TC_WEIGHTS_GATEWAY_LABELS = 1

# Modes of how to handle differences in token predictions from GatewayTokenClassifier in KeywordsFilterApproach
LOG = 'log'
DROP = 'drop'

# Log levels of filtering
FILE = 'file'
CONSOLE = 'console'

# Sampling strategies
NORMAL = 'normal'
UP_SAMPLING = 'up'
DOWN_SAMPLING = 'down'
ONLY_GATEWAYS = 'og'

# Variants how to include activity data
NOT = 'not'
DUMMY = 'dummy'
SINGLE_MASK = 'single_mask'
MULTI_MASK = 'multi_mask'
