import json
import os
import pickle

from petreader.labels import *

def format_json_file(filename: str, indent: int = 4) -> None:
    """
    reads a json file and write it back with given indent
    :param filename: filename
    :param indent: indent for formatting
    :return: None, results are written to file
    """
    with open(filename, 'r+') as file:
        result_data = json.load(file)
    os.remove(filename)
    with open(filename, 'w') as file:
        json.dump(result_data, file, indent=indent)


def save_as_pickle(obj: object, filename: str) -> None:
    """
    save an object in a pickle file dump
    :param obj: object to dump
    :param filename: target file
    :return:
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str) -> object:
    """
    load an object from a given pickle file
    :param filename: source file
    :return: loaded object
    """
    with open(os.path.abspath(filename), 'rb') as file:
        return pickle.load(file)


def goldstandards_to_json(objects: str = 'relevant') -> None:
    """
    read pickled gold standard objects and output as formatted (and filtered on objects relevant for thesis) json
    :param objects: which entities to include in output, values:
                    'relevant' -> only 'flow' and 'same gateway' for relations
                                  only 'activity', 'xor gateway', 'and gateway' and 'condition specification' for tokens
                    'all' -> all unfiltered
    :return:
    """
    def filter_goldstandard_dict(goldstandard_dict, target_object_keys):
        return {doc: {token_type: token_list for token_type, token_list in doc_dict.items()
                      if token_type.lower() in [k.lower() for k in target_object_keys]}
                for doc, doc_dict in goldstandard_dict.items()}

    relations_goldstandard = load_pickle("data/other/relations_goldstandard.pkl")
    if objects == 'relevant':
        relations_goldstandard = filter_goldstandard_dict(relations_goldstandard, [FLOW, SAME_GATEWAY])
    with open("data/other/relations_goldstandard.json", 'w') as file:
        json.dump(relations_goldstandard, file, indent=4)

    token_goldstandard = load_pickle("data/other/token_goldstandard.pkl")
    if objects == 'relevant':
        token_goldstandard = filter_goldstandard_dict(token_goldstandard, [ACTIVITY, XOR_GATEWAY, AND_GATEWAY,
                                                                           CONDITION_SPECIFICATION])
    with open("data/other/token_goldstandard.json", 'w') as file:
        json.dump(token_goldstandard, file, indent=4)


if __name__ == '__main__':
    goldstandards_to_json()