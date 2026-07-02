import re
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--prediction_file", type=str, required=True)
args = parser.parse_args()
prediction_file = args.prediction_file

def extract_answer(text, file):
    pattern = r"\(([A-D])\)" if "supergpqa" not in file else r"Answer:\s*([A-Za-z])\.?"
    
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None


df = pd.read_parquet(prediction_file).to_dict("records")

num_correct = 0
for example in df:
    gold = example['gold'][0]
    # if 'gpqa_cot' in prediction_file:
    #     gold = extract_answer(gold, prediction_file, True)
    
    prediction = example['predictions'][0]
    prediction = extract_answer(prediction, prediction_file)
    # print(gold, prediction, example['predictions'][0])
    # print("-" * 20)
    if gold and prediction:
        num_correct += (gold == prediction)

print(num_correct / len(df) * 100)