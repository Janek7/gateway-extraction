# A) CONFIG KEYS

SEED = 'general-seed'
KEYWORDS_FILTERED_APPROACH = 'keywords-filtered-approach'
BERT_MODEL_NAME = 'bert-model-name'
LABEL_SET = 'label-set'
LABEL_NUMBER = 'label-number'
NUM_LABELS = 'num-labels'
TOKEN_CLASSIFIER = 'token-classifier'
SAME_GATEWAY_CLASSIFIER = 'same-gateway-classifier'
ACTIVITY_RELATION_CLASSIFIER = 'activity-relation-classifier'
CONTEXT_LABEL_LENGTH = 'context-label-length'
SYNONYM_SAMPLES_START_NUMBER = 'synonym-samples-start-number'
EVENTUALLY_FOLLOWS_SAMPLE_LIMIT = 'eventually-follows-sample-limit'
MAX_LENGTH_RELATION_TEXT = 'max-length-relation-text'
ES_PATIENCE = 'es-patience'
MODELS = 'final-models'

# B) LABELS FOR RULE APPROACH

# available sets of keywords and contradictory keywords
LITERATURE = 'literature'  # only keyword
LITERATURE_FILTERED = 'literature_filtered'  # only keyword
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


# C) LABELS FOR SAME GATEWAY CLASSIFICATION TASK
N_GRAM = 'n_gram'
CONTEXT_NGRAM = 'context_n_gram'
CONTEXT_INDEX = 'context_index'
CONTEXT_LABELS_NGRAM = 'context_labels_n_gram'
CONTEXT_TEXT_AND_LABELS_NGRAM = 'context_text_and_labels_n_gram'

SGC_CONTEXT_LABEL_PADDING = 0
SGC_CONTEXT_LABEL_OTHER = 1
SGC_CONTEXT_LABEL_ACTIVITY = 2


# D) LABELS FOR ACTIVITY RELATION APPROACHES

# relation types
DIRECTLY_FOLLOWING = 'directly_following'
EVENTUALLY_FOLLOWING = 'eventually_following'
EXCLUSIVE = 'exclusive'
CONCURRENT = 'concurrent'

AR_LABEL_DIRECTLY_FOLLOWING = 0
AR_LABEL_EVENTUALLY_FOLLOWING = 1
AR_LABEL_EXCLUSIVE = 2
AR_LABEL_CONCURRENT = 3

# other helper keys
DOC_START = 'Document Start'
DOC_NAME = 'doc_name'
ACTIVITY_1 = 'activity_1'
ACTIVITY_2 = 'activity_2'
RELATION_TYPE = 'relation_type'
COMMENT = 'comment'

# Gateway extraction
SPLIT = 'split'
MERGE = 'merge'
XOR_GATEWAY_OPT = 'XOR Gateway (optional)'
NO_GATEWAY_RELATIONS = 'no_gateway_relations'

# Architecture variants
ARCHITECTURE_CUSTOM = 'custom'
ARCHITECTURE_CNN = 'cnn'
ARCHITECTURE_BRCNN = 'brcnn'
