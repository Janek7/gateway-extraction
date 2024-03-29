# add parent dir to sys path for import of modules
import os
import sys
# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.insert(0, parent_dir)

import logging
import os
from typing import Dict

from petreader.labels import *

from PetReader import pet_reader
from labels import SYNONYM_SAMPLES_START_NUMBER
from utils import read_keyword_synonyms, config, ROOT_DIR, load_pickle, save_as_pickle

logger = logging.getLogger('Data Augmentation')
new_sample_id = config[SYNONYM_SAMPLES_START_NUMBER]


def _generate_synonym_samples():
    """
    generate samples by replacing keywords with synonyms during copying samples
    the IDs of new samples will start at a value fixed in config.json
    :return: dictionary in format {id: {'tokens' (modified): list, 'ner-tags': list}}
    """
    # 1) load necessary data
    only_gateway_samples = [s for s in pet_reader.token_dataset.GetRandomizedSampleNumbers()
                            if f"B-{XOR_GATEWAY}" in pet_reader.token_dataset.GetSampleDictWithNerLabels(s)["ner-tags"]
                            or f"B-{AND_GATEWAY}" in pet_reader.token_dataset.GetSampleDictWithNerLabels(s)["ner-tags"]]

    only_gateway_sample_dicts = [pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number) for sample_number in
                                 only_gateway_samples]
    xor_synonyms, and_synonyms = read_keyword_synonyms()

    # 2) generate samples
    def generate_synonym_samples_dict(synonym_dict):
        synonym_sample_dicts = {}
        global new_sample_id

        for original_sample_number, original_sample_dict in zip(only_gateway_samples, only_gateway_sample_dicts):
            tokens_lower = [t.lower() for t in original_sample_dict['tokens']]
            for keyword, synonyms in synonym_dict.items():
                # check if keywords are found and real gateways in this context to prevent training on FPs
                keyword_occurences = [i for i, t in enumerate(tokens_lower) if t == keyword
                                      and original_sample_dict['ner-tags'][i].endswith("Gateway")]
                if keyword_occurences:  # possible refinement: check if keyword is not part of a larger keyword phrase
                    for synonym in synonyms:
                        tokens_modified = [t if i not in keyword_occurences else synonym for i, t in
                                           enumerate(original_sample_dict['tokens'])]
                        synonym_sample_dicts[new_sample_id] = {"original_sample_number": original_sample_number,
                                                               "tokens": tokens_modified,
                                                               "ner-tags": original_sample_dict['ner-tags']}
                        new_sample_id += 1
        return synonym_sample_dicts

    xor_synonym_samples = generate_synonym_samples_dict(xor_synonyms)
    logger.info(f"Created {len(xor_synonym_samples)} samples using XOR synonyms")
    and_synonym_samples = generate_synonym_samples_dict(and_synonyms)
    logger.info(f"Created {len(and_synonym_samples)} samples using AND synonyms")
    return {**xor_synonym_samples, **and_synonym_samples}


def get_synonym_samples() -> Dict:
    """
    read synonym samples from file; if it does not exist yet, create and save
    :return: synonym_samples dictionary
    """
    path = os.path.join(ROOT_DIR, "data/other/synonym_samples.pkl")
    if os.path.exists(path):
        logger.info(f"Reload synonym_samples from {path}")
        synonym_samples = load_pickle(path)
    else:
        logger.info(f"Create synonym_samples and save as {path}")
        synonym_samples = _generate_synonym_samples()
        save_as_pickle(synonym_samples, path)
    return synonym_samples


def get_synonyms_of_original_samples() -> Dict:
    """
    returns dictionary of original sample ids as values and related synonym sample ids as list
    :return:
    """
    synonyms = get_synonym_samples()
    synonyms_of_original_samples = {}  # dict with {original sample id: list of synonym ids}
    # record synonyms of sample ids
    for synonym_id, synonym_dict in synonyms.items():
        if synonym_dict['original_sample_number'] in synonyms_of_original_samples:
            synonyms_of_original_samples[synonym_dict['original_sample_number']].append(synonym_id)
        else:
            synonyms_of_original_samples[synonym_dict['original_sample_number']] = [synonym_id]
    # add empty lists for samples without synonyms
    for sample_id in pet_reader.token_dataset.GetRandomizedSampleNumbers():
        if sample_id not in synonyms_of_original_samples:
            synonyms_of_original_samples[sample_id] = []
    return synonyms_of_original_samples


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    x = get_synonym_samples()

