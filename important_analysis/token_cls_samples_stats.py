import os
import sys
parentdir = os.path.abspath(os.path.join(os.path.abspath(''), os.pardir))
sys.path.insert(0, parentdir)

import pandas as pd
from PetReader import pet_reader
from petreader.labels import *
from labels import *
from token_data_preparation import get_samples
from token_data_augmentation import get_synonym_samples
from utils import config


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
    docs = len(sample_dicts)
    docs_with_gateways = 0
    docs_with_xor_gateways = 0
    docs_with_and_gateways = 0

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
            docs_with_gateways += 1
        total_gateway_tokens += number_gateway_tokens
        total_other_tokens += len(sample_dict["ner-tags"]) - number_gateway_tokens

        if number_xor_gateway_tokens:
            docs_with_xor_gateways += 1
            total_xor_gateways_tokens += number_xor_gateway_tokens
        if number_and_gateway_tokens:
            docs_with_and_gateways += 1
            total_and_gateways_tokens += number_and_gateway_tokens

    # compute global stats
    avg_gateway_tokens_per_doc = round(total_gateway_tokens / docs_with_gateways, 2)
    total_share_gateway_tokens = round(total_gateway_tokens / (total_gateway_tokens + total_other_tokens), 2)

    return {"strategy": strategy,
            "docs": docs, "docs_with_gateways": docs_with_gateways, "docs_with_xor_gateways": docs_with_xor_gateways,
            "docs_with_and_gateways": docs_with_and_gateways,
            "total_gateway_tokens": total_gateway_tokens, "total_xor_gateways_tokens": total_xor_gateways_tokens,
            "total_and_gateways_tokens": total_and_gateways_tokens,
            "avg_gateway_tokens_per_doc": avg_gateway_tokens_per_doc, "total_other_tokens": total_other_tokens,
            "total_share_gateway_tokens": total_share_gateway_tokens}


rows = []
for strategy, use_synonyms in [(NORMAL, False), (UP_SAMPLING, False), (DOWN_SAMPLING, False), (ONLY_GATEWAYS, False),
                               (NORMAL, True), (ONLY_GATEWAYS, True)]:
    rows.append(analyze_samples(strategy, use_synonyms))

df = pd.DataFrame.from_dict(rows)
df.head(10)

df.to_excel("../data/paper_stats/token_cls_stats.xlsx", index=False)