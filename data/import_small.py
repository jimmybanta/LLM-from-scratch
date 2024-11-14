
import os
import json


def import_full_string():
    '''
    Imports the full string of the small dataset, from the file.

    Returns:
        all_text (str): A string containing all of the text in the dataset.
    '''

    dataset_path = os.environ['SMALL_DATASET_PATH']

    with open(dataset_path) as f:
        data = json.load(f)

    # all_text will be a massive string that contains all of the text in the dataset.
    all_text = ''

    for conversation in data:
        for item in conversation['conversations']:
            all_text += item['value'] + ' '

    return all_text
