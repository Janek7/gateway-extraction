# add parent dir to sys path for import of modules
import os
import sys

# find recursively the project root dir
parent_dir = str(os.getcwdb())
while not os.path.exists(os.path.join(parent_dir, "README.md")):
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
print(parent_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from PetReader import pet_reader
from petreader.labels import *
from labels import *
from token_approaches.token_data_preparation import get_samples
from token_approaches.token_data_augmentation import get_synonym_samples
from utils import config, ROOT_DIR


def analyze_samples(strategy, use_synonyms=False):
    print(f"Compute stats for", strategy, use_synonyms)

    # get data
    sample_numbers = get_samples(strategy, use_synonyms)
    sample_dicts = []
    synonym_samples = get_synonym_samples()
    for sample_number in sample_numbers:
        # in case sample is normal sample
        if sample_number < config[SYNONYM_SAMPLES_START_NUMBER]:
            sample_dicts.append(pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number))
        # in case sample is synonym sample
        else:
            sample_dicts.append(synonym_samples[sample_number])

    # stats to compute
    samples = len(sample_dicts)
    samples_with_gateways = 0
    samples_with_xor_gateways = 0
    samples_with_and_gateways = 0

    total_gateway_tokens = 0
    total_xor_gateways_tokens = 0
    total_and_gateways_tokens = 0
    total_other_tokens = 0

    # compute stats
    for sample_dict in sample_dicts:
        number_xor_gateway_tokens = len([tag for tag in sample_dict["ner-tags"] if tag.endswith(XOR_GATEWAY)])
        number_and_gateway_tokens = len([tag for tag in sample_dict["ner-tags"] if tag.endswith(AND_GATEWAY)])
        number_gateway_tokens = number_xor_gateway_tokens + number_and_gateway_tokens

        # fill stats
        if number_gateway_tokens:
            samples_with_gateways += 1
        total_gateway_tokens += number_gateway_tokens
        total_other_tokens += len(sample_dict["ner-tags"]) - number_gateway_tokens

        if number_xor_gateway_tokens:
            samples_with_xor_gateways += 1
            total_xor_gateways_tokens += number_xor_gateway_tokens
        if number_and_gateway_tokens:
            samples_with_and_gateways += 1
            total_and_gateways_tokens += number_and_gateway_tokens

    # compute global stats
    avg_gateway_tokens_per_doc = round(total_gateway_tokens / samples, 2)
    total_share_gateway_tokens = round(total_gateway_tokens / (total_gateway_tokens + total_other_tokens), 2)

    return {"strategy": f"{strategy}-{use_synonyms}",
            "samples": samples,
            "samples_with_gateways": samples_with_gateways,
            "samples_with_xor_gateways": samples_with_xor_gateways,
            "samples_with_and_gateways": samples_with_and_gateways,
            "total_gateway_tokens": total_gateway_tokens,
            "total_xor_gateways_tokens": total_xor_gateways_tokens,
            "total_and_gateways_tokens": total_and_gateways_tokens,
            "avg_gateway_tokens_per_doc": avg_gateway_tokens_per_doc,
            "total_other_tokens": total_other_tokens,
            "total_share_gateway_tokens": total_share_gateway_tokens}


rows = []
for strategy, use_synonyms in [(NORMAL, False), (UP_SAMPLING, False), (DOWN_SAMPLING, False), (ONLY_GATEWAYS, False),
                               (NORMAL, True), (ONLY_GATEWAYS, True)]:
    rows.append(analyze_samples(strategy, use_synonyms))

df = pd.DataFrame.from_dict(rows)
df.head(10)

df.to_excel(os.path.join(ROOT_DIR, "data/paper_stats/token_cls/token_cls_sampling_stats.xlsx"), index=False)
