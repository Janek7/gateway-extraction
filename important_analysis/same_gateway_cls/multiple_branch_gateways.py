# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
from petreader.labels import *

parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)


from PetReader import pet_reader

same_gateway_relations = []
same_gateway_relations_doc_names = []
for doc_name in pet_reader.document_names:
    doc_relations = pet_reader.relations_dataset.GetRelations(pet_reader.get_document_number(doc_name))
    doc_same_gateway_relations = doc_relations[SAME_GATEWAY]
    print(doc_name, len(doc_same_gateway_relations))
    same_gateway_relations.extend(doc_same_gateway_relations)
    same_gateway_relations_doc_names.extend([doc_name for i in range(len(doc_same_gateway_relations))])

involved_gateways = {}
for sg, doc_name in zip(same_gateway_relations, same_gateway_relations_doc_names):
    g1 = f"{doc_name}-{sg[SOURCE_SENTENCE_ID]}-{sg[SOURCE_HEAD_TOKEN_ID]}-{sg[SOURCE_ENTITY]}"
    g2 = f"{doc_name}-{sg[TARGET_SENTENCE_ID]}-{sg[TARGET_HEAD_TOKEN_ID]}-{sg[TARGET_ENTITY]}"
    if g1 in involved_gateways:
        involved_gateways[g1] += 1
    else:
        involved_gateways[g1] = 1
    if g2 in involved_gateways:
        involved_gateways[g2] += 1
    else:
        involved_gateways[g2] = 1

involved_gateways = dict(sorted(involved_gateways.items(), key=lambda item: item[1], reverse=True))
for g, number in involved_gateways.items():
    print(g, number)
