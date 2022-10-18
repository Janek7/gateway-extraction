import tensorflow as tf
import transformers
from petreader.labels import *
from labels import *
from PetReader import pet_reader
import logging
from typing import Tuple


logger = logging.getLogger('data preparation')


def create_token_classification_dataset(tokenizer: transformers.BertTokenizerFast, dev_share: int = 0.1,
                                        batch_size: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    create the dataset for token classification with huggingface transformers bert like models
    split into train and dev/validation by the given share in dev_share
    tokens are labeled into XOR, AND and OTHER. Additionally a label for bert specific tokens such as CLS, SEP and PAD
    :param tokenizer: instance of subclass of transformers.BertTokenizerFast
    :param dev_share: share of validation/development dataset
    :param batch_size: apply batching to size if given
    :return: tf.data.Dataset instance
    """
    sample_numbers = pet_reader.token_dataset.GetRandomizedSampleNumbers()
    sample_dicts = [pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number)
                    for sample_number in sample_numbers]
    sample_sentences = [sample_dict['tokens'] for sample_dict in sample_dicts]

    # 1) tokenize all samples in batch mode in order to retrieve token ids and attention masks
    all_tokens = tokenizer(sample_sentences, is_split_into_words=True, padding=True, return_tensors='tf')
    max_sentence_length = all_tokens['input_ids'].shape[1]

    # 2) transform NER token tags into labels for classification
    all_sample_labels = []
    for i, sample_number in enumerate(sample_numbers):
        sample_dict = pet_reader.token_dataset.GetSampleDictWithNerLabels(sample_number)
        # transformer_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][i])
        # tokenize again every single sample to get access to .word_ids()
        tokenization = tokenizer(sample_dict['tokens'], is_split_into_words=True,
                                 padding='max_length', max_length=max_sentence_length, return_tensors='tf')
        sample_tokens = tokenizer.convert_ids_to_tokens(tokenization['input_ids'][0])

        sample_labels = []
        # word index necessary, because one token in PET could be splitted into multiple tokens with tokenizer
        # multiple tokens have all the same word_id -> allows retrieval of the same one NER label from PET tokens
        for token, word_index in zip(sample_tokens, tokenization.word_ids()):
            # set special class for special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                sample_labels.append(LABEL_OUT_OF_SCOPE)
            else:
                token_tag = sample_dict['ner-tags'][word_index]
                if token_tag.endswith(XOR_GATEWAY):
                    sample_labels.append(LABEL_XOR)
                elif token_tag.endswith(AND_GATEWAY):
                    sample_labels.append(LABEL_AND)
                else:
                    sample_labels.append(LABEL_OTHER)

        all_sample_labels.append(sample_labels)

    # create tensorflow dataset and split into train and dev
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': all_tokens['input_ids'],
                                                   'attention_mask': all_tokens['attention_mask']},
                                                  tf.constant(all_sample_labels)))
    number_samples = all_tokens['input_ids'].shape[0]
    val_samples = round(number_samples * dev_share)
    train_dataset = dataset.take(val_samples)
    val_dataset = dataset.skip(val_samples)

    logger.info(f"Created token classification dataset of shape {all_tokens['input_ids'].shape} splitted into "
                f"{1-dev_share}/{dev_share} -> {number_samples - val_samples}/{val_samples} (train/dev)")
    if batch_size:
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = create_token_classification_dataset(tokenizer, batch_size=8)
