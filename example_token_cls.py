from petreader.TokenClassification import TokenClassification

dataset = TokenClassification()

print(dir(dataset))  # ['GetANDGateways', 'GetActivities', 'GetActivityData', 'GetActors', 'GetConditionSpecifications', 'GetDocumentANDGateways', 'GetDocumentANDGatewaysIndexes', 'GetDocumentActivities', 'GetDocumentActivitiesIndexes', 'GetDocumentActivityData', 'GetDocumentActivityDataIndexes', 'GetDocumentActors', 'GetDocumentActorsIndexes', 'GetDocumentConditionSpecifications', 'GetDocumentConditionSpecificationsIndexes', 'GetDocumentFurtherSpecifications', 'GetDocumentFurtherSpecificationsIndexes', 'GetDocumentName', 'GetDocumentNames', 'GetDocumentText', 'GetDocumentXORGateways', 'GetDocumentXORGatewaysIndexes', 'GetFurtherSpecifications', 'GetNerTagIDs', 'GetNerTagId', 'GetNerTagLabel', 'GetNerTagLabels', 'GetPrefixAndLabel', 'GetRandomizedSampleNumbers', 'GetSampleDict', 'GetSampleDictWithNerLabels', 'GetSampleEntities', 'GetSampleEntitiesIndexes', 'GetSampleEntitiesWithTokenIds', 'GetSampleListOfEntities', 'GetSentenceID', 'GetTokens', 'GetXORGateways', '_NER_TAGS', '_NER_TAGS_ID_TO_LABEL_MAP', '_NER_TAGS_LABEL_TO_ID_MAP', '_TokenClassification__create_n_sample_document_ids', '_TokenClassification__documents_ids', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_create_process_element_indexes_list', '_create_process_element_list', '_get_process_element_indexes_list', '_get_process_element_list', '_process_element_indexes_list', '_process_element_list', '_remove_B_I_prefix', 'dataset', 'get_n_sample_of_a_document']

print(dataset.GetDocumentNames())  # ['doc-10.5', 'doc-3.3', 'doc-8.2', 'doc-6.1', 'doc-6.3', 'doc-9.2', 'doc-10.9', 'doc-10.14', 'doc-3.1', 'doc-3.5', 'doc-6.4', 'doc-3.8', 'doc-3.7', 'doc-10.3', 'doc-10.6', 'doc-6.2', 'doc-1.2', 'doc-10.8', 'doc-2.1', 'doc-5.3', 'doc-10.4', 'doc-5.1', 'doc-9.5', 'doc-1.4', 'doc-10.11', 'doc-4.1', 'doc-10.10', 'doc-9.4', 'doc-1.1', 'doc-1.3', 'doc-2.2', 'doc-9.3', 'doc-3.6', 'doc-5.4', 'doc-10.7', 'doc-5.2', 'doc-7.1', 'doc-10.13', 'doc-10.1', 'doc-3.2', 'doc-10.2', 'doc-10.12', 'doc-9.1', 'doc-8.3', 'doc-8.1']

print(dataset.GetDocumentName(0))  # doc-1.1

print(dataset.GetTokens(0))  # ['A', 'small', 'company', 'manufactures', 'customized', 'bicycles', '.']

print(dataset.GetNerTagLabels(0))  # ['O', 'O', 'O', 'O', 'O', 'O', 'O']

# return the ner tag name of a given ner tag id
print(dataset.GetNerTagLabel(0))  # 0

print(dataset.GetNerTagLabel(1))  # B-Actor

print(dataset.GetNerTagLabel(2))  # I-Actor

print(dataset.GetDocumentActivities('doc-1.1')) # list of [sentence-number: list of activities]
# [[], [['receives']], [['reject'], ['accept']], [], [['informed']], [['processes'], ['checks']], [['reserved']], [['back-ordered']], [], [['prepares']], [['assembles']], [['ships']]]
