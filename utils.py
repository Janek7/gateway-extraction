import json
import os
import pickle
from typing import List, Tuple, Dict
import logging

from petreader.labels import *
from labels import *

logger = logging.getLogger('utilities')
logging.basicConfig(level=logging.INFO)


def read_and_set_keywords(keywords: str) -> Tuple[List[str], List[str]]:
    """
    load and return key word lists based on passed variant
    :return: two list
    """
    logger.info(f"Load keywords '{keywords}' ...")
    if keywords == LITERATURE:
        # based on key words proposals of Ferreira et al. 2017
        with open('data/keywords/literature_xor.txt') as file:
            xor_keywords = file.read().splitlines()

        with open('data/keywords/literature_and.txt') as file:
            and_keywords = file.read().splitlines()

    elif keywords == GOLD:
        from PetReader import pet_reader
        xor_keywords = pet_reader.xor_key_words_gold
        and_keywords = pet_reader.and_key_words_gold

    elif keywords == OWN:
        raise NotImplementedError("Own keywords not implemented yet")

    xor_keywords.sort()
    and_keywords.sort()

    logger.info(f"Loaded {len(xor_keywords)} XOR and {len(and_keywords)} AND keywords ({keywords})")
    logger.info(f"Used XOR keywords: {xor_keywords}")
    logger.info(f"Used AND keywords: {and_keywords}")

    return xor_keywords, and_keywords


def read_contradictory_gateways() -> List[Tuple[List[str], List[str]]]:
    """
    read pairs of contradictory exclusive gateway key words from file
    sort to prefer longer matching phrases during search
    :return: list of pairs
    """
    with open('data/keywords/contradictory_gateways_gold.txt') as file:
        contradictory_gateways = [[x.split(" ") for x in l.strip().split(";")] for l in file.readlines()]
        logger.info(f"Loaded {len(contradictory_gateways)} pairs of contradictory keywords")
        contradictory_gateways.sort(key=lambda pair: len(pair[0]) + len(pair[1]), reverse=True)
        return contradictory_gateways


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


def read_json_to_dict(filename: str) -> Dict:
    with open(filename, 'r') as file:
        return json.load(file)


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
