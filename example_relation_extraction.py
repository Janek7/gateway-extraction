from petreader.RelationsExtraction import RelationsExtraction
from petreader.labels import FLOW, SAME_GATEWAY

dataset = RelationsExtraction()
doc_id = 1

# document structure
print(" document structure ".center(50, '*'))
print(dataset.GetDocument(doc_id))
print(dataset.GetSentencesWithIdsAndNerTagLabels(doc_id))  # [[('A', 0, 'B-Actor'), ('customer', 1, 'I-Actor'), ('brings', 2, 'B-Activity'), ('in', 3, 'I-Activity'), ('a', 4, 'B-Activity Data'), ('defective', 5, 'I-Activity Data'), ('computer', 6, 'I-Activity Data'), ('and', 7, 'O'), ('the', 8, 'B-Actor'), ('CRS', 9, 'I-Actor'), ('checks', 10, 'B-Activity'), ('the', 11, 'B-Activity Data'), ('defect', 12, 'I-Activity Data'), ('and', 13, 'O'), ('hands', 14, 'B-Activity'), ('out', 15, 'I-Activity'), ('a', 16, 'B-Activity Data'), ('repair', 17, 'I-Activity Data'), ('cost', 18, 'I-Activity Data'), ('calculation', 19, 'I-Activity Data'), ('back', 20, 'O'), ('.', 21, 'O')], [('If', 0, 'B-XOR Gateway'), ('the', 1, 'B-Condition Specification'), ('customer', 2, 'I-Condition Specification'), ('decides', 3, 'I-Condition Specification'), ('that', 4, 'I-Condition Specification'), ('the', 5, 'I-Condition Specification'), ('costs', 6, 'I-Condition Specification'), ('are', 7, 'I-Condition Specification'), ('acceptable', 8, 'I-Condition Specification'), (',', 9, 'O'), ('the', 10, 'O'), ('process', 11, 'O'), ('continues', 12, 'O'), (',', 13, 'O'), ('otherwise', 14, 'B-XOR Gateway'), ('she', 15, 'B-Actor'), ('takes', 16, 'B-Activity'), ('her', 17, 'B-Activity Data'), ('computer', 18, 'I-Activity Data'), ('home', 19, 'B-Further Specification'), ('unrepaired', 20, 'I-Further Specification'), ('.', 21, 'O')], [('The', 0, 'O'), ('ongoing', 1, 'O'), ('repair', 2, 'O'), ('consists', 3, 'O'), ('of', 4, 'O'), ('two', 5, 'O'), ('activities', 6, 'O'), (',', 7, 'O'), ('which', 8, 'O'), ('are', 9, 'O'), ('executed', 10, 'O'), (',', 11, 'O'), ('in', 12, 'O'), ('an', 13, 'O'), ('arbitrary', 14, 'O'), ('order', 15, 'O'), ('.', 16, 'O')], [('The', 0, 'O'), ('first', 1, 'O'), ('activity', 2, 'O'), ('is', 3, 'O'), ('to', 4, 'O'), ('check', 5, 'B-Activity'), ('and', 6, 'O'), ('repair', 7, 'B-Activity'), ('the', 8, 'B-Activity Data'), ('hardware', 9, 'I-Activity Data'), (',', 10, 'O'), ('whereas', 11, 'B-AND Gateway'), ('the', 12, 'O'), ('second', 13, 'O'), ('activity', 14, 'O'), ('checks', 15, 'B-Activity'), ('and', 16, 'O'), ('configures', 17, 'B-Activity'), ('the', 18, 'B-Activity Data'), ('software', 19, 'I-Activity Data'), ('.', 20, 'O')], [('After', 0, 'O'), ('each', 1, 'O'), ('of', 2, 'O'), ('these', 3, 'O'), ('activities', 4, 'O'), (',', 5, 'O'), ('the', 6, 'B-Activity Data'), ('proper', 7, 'I-Activity Data'), ('system', 8, 'I-Activity Data'), ('functionality', 9, 'I-Activity Data'), ('is', 10, 'O'), ('tested', 11, 'B-Activity'), ('.', 12, 'O')], [('If', 0, 'B-XOR Gateway'), ('an', 1, 'B-Condition Specification'), ('error', 2, 'I-Condition Specification'), ('is', 3, 'I-Condition Specification'), ('detected', 4, 'I-Condition Specification'), ('another', 5, 'B-Activity Data'), ('arbitrary', 6, 'I-Activity Data'), ('repair', 7, 'I-Activity Data'), ('activity', 8, 'I-Activity Data'), ('is', 9, 'O'), ('executed', 10, 'B-Activity'), (',', 11, 'O'), ('otherwise', 12, 'B-XOR Gateway'), ('the', 13, 'O'), ('repair', 14, 'O'), ('is', 15, 'O'), ('finished', 16, 'O'), ('.', 17, 'O')]]

# relations

relations = dataset.GetRelations(doc_id)
flow_relations, same_gateway_relations = relations[FLOW], relations[SAME_GATEWAY]

print(" flow relations ".center(50, '*'))
for flow_relation in flow_relations[:1]:
    for key, value in flow_relation.items():
        print(f"{key}: {value}")
    print()

print()

print(" same gateway relations ".center(50, '*'))
for same_gateway_relation in same_gateway_relations:
    for key, value in same_gateway_relation.items():
        print(f"{key}: {value}")
    print()