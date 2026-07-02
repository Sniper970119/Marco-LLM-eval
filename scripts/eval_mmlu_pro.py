import re
import pandas as pd
from argparse import ArgumentParser
import os
import glob


parser = ArgumentParser()
parser.add_argument("--prediction_file", type=str, required=True)
args = parser.parse_args()
prediction_file = args.prediction_file

def extract_answer(text):
    pattern = r"\(([A-Z])\)"
    
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None

def check_single_file(input_file):
    df = pd.read_parquet(input_file).to_dict("records")

    num_correct = 0
    cnt = 0
    for example in df:
        gold = example['gold'][0].strip("()")
        
        prediction = example['predictions'][0].strip().split("Answer the following multiple choice question. Think step by step before answering.")[0].strip()
        prediction = prediction.split("You are an AI assistant")[0].strip()
        prediction = extract_answer(prediction)
        if not prediction:
            cnt += 1
            # print(gold, '\t', example['predictions'][0].strip())
        if gold and prediction:
            num_correct += (gold == prediction)

    print(num_correct / len(df) * 100, cnt)

if os.path.isfile(prediction_file):
    check_single_file(prediction_file)
else:
    files = glob.glob(prediction_file + "/*")
    for file in files:
        print(file.split("/")[-1].split("|")[1].split("_")[-1])
        check_single_file(file)

