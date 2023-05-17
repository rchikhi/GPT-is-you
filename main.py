import pandas as pd
from collections import defaultdict
import re
import os


GENERAL_WA_MULTI_SEARCH_PATTERN = (
    r"\[(\d{1,2}\/\d{1,2}\/\d{2}), (\d{1,2}:\d{2}:\d{2} [APM]{2})\] (.*?): (.*)"
)
LINE_SPLIT_DELIMITER = "\n"
IGNORE_LIST = ["omitted", "Missed voice call"]
RESPONDER = "Joey Aramouni"


def text_to_dictionary(text, response):
    """
    We convert a whatsapp chat into a prompt and
    response dataframe for the purposes of finetuning.
    :param text:
    :param response:
    :return:
    """
    # convert from bytes to string
    if isinstance(text, (bytes, bytearray)):
        text = text.decode()

    text_list = text.split("\n")[1:]
    result_dict, count, prev_author = defaultdict(dict), 0, ""
    for line in text_list:
        search_pattern = re.search(GENERAL_WA_MULTI_SEARCH_PATTERN, line)
        if search_pattern is not None:
            author = search_pattern.group(3)
            message = search_pattern.group(4).replace('"', "")
        else:
            continue
        if author != response:
            if author == prev_author:
                prev = result_dict[count]["prompt"][:-7]
                result_dict[count]["prompt"] = f"{prev}. {message}\n\n###\n\n"
            else:
                count += 1
                result_dict[count].update({"prompt": message + "\n\n###\n\n"})

        else:
            if author == prev_author:
                prev = result_dict[count]["completion"][:-4]
                result_dict[count]["completion"] = f"{prev}. {message} ###"
            else:
                result_dict[count].update({"completion": message + " ###"})

        prev_author = author
    return result_dict


def parse_whatsapp_text_into_dataframe(raw_text, responder):
    result_dict = text_to_dictionary(raw_text, responder)
    df = pd.DataFrame.from_dict(result_dict).T[["prompt", "completion"]]
    return df


def converter(
    filepath: str,
    responder: str,
) -> pd.DataFrame:
    """
    Turn whatsapp chat data into a format
    that can be trained by openai's fine tuning api.
    :param file: Path to file we want to convert
    :param responder: The person to be labelled as the responder
    :return: a parsed pandas dataframe
    """

    with open(filepath, "r") as fp:
        text = fp.read()

    df = parse_whatsapp_text_into_dataframe(text, responder)
    df = df.dropna()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove prompts and completion with attachement.
    :param df: parsed pandas dataframe with attachement
    :return: parsed pandas dataframe without attachement
    """

    df = df.copy()
    df["includes_attachment"] = df.apply(
        lambda x: True
        if any(
            keyword in x[col]
            for keyword in IGNORE_LIST
            for col in ["prompt", "completion"]
        )
        else False,
        axis=1,
    )
    return (
        df.loc[lambda x: x["includes_attachment"] == False]
        .drop("includes_attachment", axis=1)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    df_final = pd.DataFrame()
    for file in os.listdir("chats"):
        print(f"Preprocessing conv with {file.split('.')[0]}")
        df = converter(f"./chats/{file}", RESPONDER)
        df_final = df if not len(df_final) else pd.concat([df_final, df])
    df_final = clean(df_final)
    df_final.to_json(
        f"output/output.json", lines=True, orient="records", force_ascii=False
    )
