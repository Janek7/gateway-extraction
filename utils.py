import json
import os
import pickle


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


def save_as_pickle(obj, filename):
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


def load_pickle(filename):
    """
    load an object from a given pickle file
    :param filename: source file
    :return: loaded object
    """
    with open(os.path.abspath(filename), 'rb') as file:
        return pickle.load(file)
