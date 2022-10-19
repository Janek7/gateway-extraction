# set of keywords available
LITERATURE = 'literature'
GOLD = 'gold'
OWN = 'own'

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
GTC_LABEL_OUT_OF_SCOPE = 0
GTC_LABEL_OTHER = 1
GTC_LABEL_XOR = 2
GTC_LABEL_AND = 3
GTC_LABELS = [GTC_LABEL_OUT_OF_SCOPE, GTC_LABEL_OTHER, GTC_LABEL_XOR, GTC_LABEL_AND]
