from petreader.labels import *

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

# labels for token classification task
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
