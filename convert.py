import argparse
import datetime
import pandas as pd
from collections import defaultdict
import re
from pathlib import Path
GENERAL_WA_MULTI_SEARCH_PATTERN = r'\d?\d\/\d?\d\/\d?\d?\d\d, \d\d:\d\d:?\d?\d?\s?-? (.*?):(.*)'
LINE_SPLIT_DELIMITER = "\n"


def text_to_dictionary(text, prompt, response):
    """
    We convert a whatsapp chat into a prompt and
    response dataframe for the purposes of finetuning.
    :param text:
    :param prompt:
    :param response:
    :return:
    """
    # convert from bytes to string
    if isinstance(text, (bytes, bytearray)):
        text = text.decode()

    text_list = text.split('\n')[1:]
    result_dict, count, prev_author = defaultdict(dict), 0, ''
    for ix, line in enumerate(text_list):
        search_pattern = re.search(GENERAL_WA_MULTI_SEARCH_PATTERN, line)
        if search_pattern is not None:
            author = search_pattern.group(1)
            message = search_pattern.group(2).replace('"','')
        else:
            continue
        if author == prompt:
            if author == prev_author:
                prev = result_dict[count]['prompt'][:-7]
                result_dict[count]['prompt'] = f"{prev}. {message}\n\n###\n\n"
            else:
                count += 1
                result_dict[count].update({'prompt': message + "\n\n###\n\n"})

        elif author == response:
            if author == prev_author:
                prev = result_dict[count]['completion'][:-4]
                result_dict[count]['completion'] = f"{prev}. {message} ###"
            else:
                result_dict[count].update({'completion': message + " ###"})

        prev_author = author
    return result_dict

def parse_whatsapp_text_into_dataframe(raw_text, prompter, responder):
    result_dict = text_to_dictionary(raw_text, prompter, responder)

    df = pd.DataFrame.from_dict(result_dict).T[['prompt', 'completion']]
    return df


def converter(
        filepath: str,
        prompter: str,
        responder: str,
) -> pd.DataFrame:
    """
    Turn whatsapp chat data into a format
    that can be trained by openai's fine tuning api.
    :param file: Path to file we want to convert
    :param prompter: The person to be labelled as the prompter
    :param responder: The person to be labelled as the responder
    :return: a parsed pandas dataframe
    """

    with open(filepath, 'r') as fp:
        text = fp.read()

    df = parse_whatsapp_text_into_dataframe(text, prompter, responder)
    df = df.dropna()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process your whatsapp chat data.')
    parser.add_argument('path', type=str, help='Path to file')
    parser.add_argument('prompter', type=str, help='Name of Prompter')
    parser.add_argument('responder', type=str, help='Name of Responder')
    parser.add_argument('-filename', type=str, help='Destination filename')

    args = parser.parse_args()
    path = args.path
    prompter = args.prompter
    responder = args.responder
    filename = args.filename

    save_file = filename if filename else datetime.datetime.now()
    converter(path, prompter, responder).to_json(f'output_{save_file}.json',lines=True,orient='records', force_ascii=False)


