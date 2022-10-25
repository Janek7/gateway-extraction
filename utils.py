import json
import os


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
