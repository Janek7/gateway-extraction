import argparse
import itertools
import json
import os
import pickle
import re
import datetime
from typing import List, Tuple, Dict, IO
import logging
import random

import numpy as np
from petreader.labels import *
from labels import *

logger = logging.getLogger('Utilities')
logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # project root path


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


class GatewayExtractionException(Exception):
    """
    custom exception
    """
    def __init__(self, message):
        super().__init__(message)


def debugging(func):
    """
    empty decorator function to mark functions for debugging use only
    :param func: function
    :return: unmodified function
    """
    return func


def get_keywords(keywords: str, token_flattened: bool = False) -> Tuple[List[str], List[str]]:
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


def read_keyword_synonyms() -> Tuple[Dict, Dict]:
    """
    read dictionaries of keyword synonyms from one synonym file
    :return: two dicts with keywords as key and a list of synonyms as values
    """
    with open(os.path.join(ROOT_DIR, 'data/keywords/synonyms.json')) as file:
        synonym_dict = json.load(file)
    return synonym_dict["xor"], synonym_dict["and"]


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


def get_contradictory_gateways(contradictory_keywords: str) \
        -> List[Tuple[List[str], List[str]]]:
    """
    read pairs of contradictory exclusive gateway key words from file
    sort to prefer longer matching phrases during search
    :param contradictory_keywords: flag/variant which contradictory keyword pairs to use; available: custom, gold
    :return: list of pairs
    """
    def process_lines(lines):
        contradictory_gateways = [tuple([x.split(" ") for x in l.strip().split(";")]) for l in lines]
        logger.info(f"Loaded {len(contradictory_gateways)} gold pairs of contradictory keywords")
        contradictory_gateways.sort(key=lambda pair: len(pair[0]) + len(pair[1]), reverse=True)
        return contradictory_gateways

    if contradictory_keywords == GOLD:
        path = os.path.join(ROOT_DIR, "data/keywords/contradictory_gateways_gold.txt")
        if not os.path.exists(path):
            write_gold_contradictory_keywords_to_files()
        with open(path) as file:
            return process_lines(file.readlines())

    elif contradictory_keywords == CUSTOM:
        with open(os.path.join(ROOT_DIR, "data/keywords/contradictory_gateways.txt")) as file:
            return process_lines(file.readlines())


def write_gold_contradictory_keywords_to_files() -> None:
    """
    write gold contradictory keyword pairs to file
    :return:
    """
    from PetReader import pet_reader
    with open(os.path.join(ROOT_DIR, 'data/keywords/contradictory_gateways_gold.txt'), 'w') as file:
        contradictory_gateways = pet_reader.extract_gold_contradictory_keywords()
        for pair in contradictory_gateways:
            file.write("%s\n" % f"{''.join(pair[0])};{''.join(pair[1])}")
        logger.info(f"Wrote {len(contradictory_gateways)} gold pairs of contradictory keywords")
        return contradictory_gateways


def load_loop_flows() -> Dict[str, Dict]:
    """
    load flows that cause loops from file
    :return: dictionary with sequence flow that causes loop for each document where a loop is contained
    """
    with open(os.path.join(ROOT_DIR, 'data/activity_relation/loops.json'), 'r') as file:
        loop_flows = json.load(file)["loop_flows"]
        for doc_name, loop_flow in loop_flows.items():
            loop_flows[doc_name][SOURCE] = tuple(loop_flows[doc_name][SOURCE])
            loop_flows[doc_name][TARGET] = tuple(loop_flows[doc_name][TARGET])
        return loop_flows


def load_activity_relation_test_docs() -> List[str]:
    """
    reads list of test docs stored in json file
    predefined test set of random docs (about 10% of relations of the whole set)
    :return: list of doc names
    """
    with open(os.path.join(ROOT_DIR, "data/activity_relation/test_docs.json", "r")) as file:
        return json.load(file)["test_docs"]


def load_activity_relation_black_list_docs() -> List[str]:
    """
    reads list of black list stored in json file
    documents contain black list with very complicated nested structures where data generation fails
    :return: list of doc names
    """
    with open(os.path.join(ROOT_DIR, "data/activity_relation/black_list.json"), "r") as file:
        return json.load(file)["black_list_docs"]


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


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


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


def generate_args_logdir(args: argparse.Namespace, script_name: str = None) -> str:
    """
    generates a log directory for saving training results based on filename, date und arguments
    :param script_name: script name for which to generate logdir for
    :param args: args
    :return: logdir as string
    """
    if not script_name:
        script_name = os.path.basename(globals().get(script_name, "notebook"))
    return os.path.join(ROOT_DIR, "data/logs", "{}-{}-{}".format(
        script_name,
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))


def save_args_to_file(args, log_dir):
    """
    save arguments line by line to given filename
    :param args: namespace arguments
    :param log_dir: name of log folder
    :return:
    """
    log_dir = os.path.join(ROOT_DIR, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.txt"), 'w') as file:
        longest_key = max([len(k) for k in vars(args).keys()])
        file.write(f"{f'date:'.ljust(longest_key + 2)}{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        for k, v in sorted(vars(args).items()):
            file.write(f"{f'{k}:'.ljust(longest_key + 2)}{v}\n")


def flatten_list(two_dim_list: List[List]) -> List:
    """
    flattens a two dimensional list
    :param two_dim_list: nested list
    :return: not nested list
    """
    return [item for sublist in two_dim_list for item in sublist]


# save in variable, because it has to be always called random.seed(...) before every random.X call
CURRENT_USED_SEED = 0


def set_seeds(seed: int, caller: str = None) -> None:
    """
    set tensorflow seeds
    :param seed: seed
    :param caller: where seed change was called, optional
    :return:
    """
    logger.info(f"Set seeds to {seed} (caller: {caller})")
    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    global CURRENT_USED_SEED
    CURRENT_USED_SEED = seed


def get_seed_list(seed_param_str: str):
    """
    create a list of seeds in the range between the start and end given in the string
    :param seed_param_str: start/end seed -> format: "0-10"
    :return: list of seeds
    """
    split = seed_param_str.split("-")
    start, end = int(split[0]), int(split[1])
    return list(range(start, end+1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
