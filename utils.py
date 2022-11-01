import itertools
import json
import os
import pickle
from typing import List, Tuple, Dict, IO
import logging

from petreader.labels import *
from labels import *

logger = logging.getLogger('Utilities')
logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # project root path
SEED_SET_DURING_SESSION = False


def read_config() -> Dict:
    """
    read config from file into dict
    :return: config as Dict
    """
    with open(os.path.join(ROOT_DIR, "config.json"), 'r') as file:
        config = json.load(file)
    config[KEYWORDS_FILTERED_APPROACH][NUM_LABELS] = 4 if config[KEYWORDS_FILTERED_APPROACH][LABEL_SET] == FILTERED else 9
    logger.info(f"Loaded config: {str(config)}")
    return config


config = read_config()


def read_keywords(keywords: str, token_flattened: bool = False) -> Tuple[List[str], List[str]]:
    """
    load and return key word lists based on passed variant
    :param keywords: string constant which set to load
    :param token_flattened: flag if keywords should be splitted into tokens
    :return: two list
    """
    logger.info(f"Load keywords '{keywords}' ...")
    if keywords == LITERATURE:
        # based on key words proposals of Ferreira et al. 2017
        with open(os.path.join(ROOT_DIR, 'data/keywords/literature_xor.txt')) as file:
            xor_keywords = file.read().splitlines()

        with open(os.path.join(ROOT_DIR, 'data/keywords/literature_and.txt')) as file:
            and_keywords = file.read().splitlines()

    elif keywords == LITERATURE_FILTERED:
        # based on key words proposals of Ferreira et al. 2017 (filtered
        with open(os.path.join(ROOT_DIR, 'data/keywords/literature_xor_filtered.txt')) as file:
            xor_keywords = file.read().splitlines()

        with open(os.path.join(ROOT_DIR, 'data/keywords/literature_and_filtered.txt')) as file:
            and_keywords = file.read().splitlines()

    elif keywords == GOLD:
        from PetReader import pet_reader
        xor_keywords = pet_reader.xor_key_words_gold
        and_keywords = pet_reader.and_key_words_gold

    elif keywords == CUSTOM:

        def read_custom_file(file: IO) -> List[str]:
            """
            read file in csv format -> cols: phrase;source;keep
            include just the phrases that have third column on 'keep' -> omit 'drop' phrases
            :param file: file object
            :return:
            """
            return [kw_phrase.split(";")[0] for kw_phrase in file.read().splitlines()
                    if kw_phrase.split(";")[2] == "keep"]

        with open(os.path.join(ROOT_DIR, 'data/keywords/custom_xor.csv')) as file:
            xor_keywords = read_custom_file(file)

        with open(os.path.join(ROOT_DIR, 'data/keywords/custom_and.csv')) as file:
            and_keywords = read_custom_file(file)

    if token_flattened:
        xor_keywords = list(set(itertools.chain(*[keyword.split(" ") for keyword in xor_keywords])))
        and_keywords = list(set(itertools.chain(*[keyword.split(" ") for keyword in and_keywords])))

    xor_keywords.sort()
    and_keywords.sort()

    logger.info(f"Loaded {len(xor_keywords)} XOR and {len(and_keywords)} AND keywords ({keywords})")
    logger.info(f"Used XOR keywords: {xor_keywords}")
    logger.info(f"Used AND keywords: {and_keywords}")

    return xor_keywords, and_keywords


def write_gold_keywords_to_files() -> None:
    """
    write gold keywords to files
    :return:
    """
    from PetReader import pet_reader

    with open(os.path.join(ROOT_DIR, 'data/keywords/gold_xor.txt'), 'w') as file:
        for keyword in sorted(pet_reader.xor_key_words_gold):
            file.write("%s\n" % keyword)

    with open(os.path.join(ROOT_DIR, 'data/keywords/gold_and.txt'), 'w') as file:
        for keyword in sorted(pet_reader.and_key_words_gold):
            file.write("%s\n" % keyword)


def read_contradictory_gateways() -> List[Tuple[List[str], List[str]]]:
    """
    read pairs of contradictory exclusive gateway key words from file
    sort to prefer longer matching phrases during search
    :return: list of pairs
    """
    with open(os.path.join(ROOT_DIR, 'data/keywords/contradictory_gateways_gold.txt')) as file:
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

    relations_goldstandard = load_pickle(os.path.join(ROOT_DIR, "data/other/relations_goldstandard.pkl"))
    if objects == 'relevant':
        relations_goldstandard = filter_goldstandard_dict(relations_goldstandard, [FLOW, SAME_GATEWAY])
    with open(os.path.join(ROOT_DIR, "data/other/relations_goldstandard.json"), 'w') as file:
        json.dump(relations_goldstandard, file, indent=4)

    token_goldstandard = load_pickle(os.path.join(ROOT_DIR, "data/other/token_goldstandard.pkl"))
    if objects == 'relevant':
        token_goldstandard = filter_goldstandard_dict(token_goldstandard, [ACTIVITY, XOR_GATEWAY, AND_GATEWAY,
                                                                           CONDITION_SPECIFICATION])
    with open(os.path.join(ROOT_DIR, "data/other/token_goldstandard.json"), 'w') as file:
        json.dump(token_goldstandard, file, indent=4)


def set_seeds(seed: int, overwrite=False) -> None:
    """
    set tensorflow seeds
    :param seed: seed
    :param overwrite: flag if overwrite of seed should be allowed; only usage in following initial set_seeds intended
    :return:
    """
    if not SEED_SET_DURING_SESSION or overwrite:
        logger.info(f"Set seeds to {seed} (overwrite={overwrite})")
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.compat.v1.set_random_seed(seed)

    if SEED_SET_DURING_SESSION and overwrite:
        logger.warning("utils.set_seeds overwrites already set seed")


set_seeds(config[SEED], overwrite=True)  # set always, will maybe overwritten by seed of args


if __name__ == '__main__':
    # goldstandards_to_json()

    # xor_keywords, and_keywords = read_keywords(CUSTOM)
    # print(xor_keywords)
    # print(and_keywords)

    print(config[KEYWORDS_FILTERED_APPROACH][BERT_MODEL_NAME])
