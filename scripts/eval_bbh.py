import re
import pandas as pd
from argparse import ArgumentParser
import glob

BBH_SUBSETS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]

parser = ArgumentParser()
parser.add_argument("--prediction_dir", type=str, required=True)
args = parser.parse_args()
prediction_dir = args.prediction_dir

def extract_answer_boolean_expressions(text):
    pattern = r"answer\s+is\s+(True|False)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

def extract_answer_causal_judgement(text):
    pattern = r"answer\s+is\s+(Yes|No)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

def extract_answer_date_understanding(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_disambiguation_qa(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_dyck_languages(text):
    # < [ < [ { < [ ] < { } > > } ] > { { ( ) } { < [ < > ] > }
    pattern = r"answer is ([<\[\{}\]>() ]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_answer_formal_fallacies(text):
    pattern = r"answer\s+is\s+(valid|invalid)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def extract_answer_geometric_shapes(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_hyperbaton(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_logical_deduction(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_movie_recommendation(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_multistep_arithmetic_two(text):
    pattern = r"So the answer is (-?\d+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)  
    return None

def extract_answer_navigate(text):
    pattern = r"answer\s+is\s+(Yes|No)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

def extract_answer_object_counting(text):
    pattern = r"So the answer is (-?\d+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)  
    return None

def extract_answer_penguins_in_a_table(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_reasoning_about_colored_objects(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_ruin_names(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_salient_translation_error_detection(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_snarks(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_sports_understanding(text):
    pattern = r"answer\s+is\s+(yes|no)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_answer_temporal_sequences(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def tracking_shuffled_objects(text):
    pattern = r"\(([A-Za-z])\)"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    return None

def extract_answer_web_of_lies(text):
    pattern = r"answer\s+is\s+(Yes|No)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

def extract_answer_word_sorting(text):
    patterns = [
        r"answer is ([a-z\s]+)(?:\.|$)",
        r"answer is ([a-zA-Z\s]+)(?:\.|$)",
        r"answer is ([\w\s]+)(?:\.|$)",
        r"So the answer is ([^.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    return None

eval_funcs = {
    "boolean_expressions": extract_answer_boolean_expressions,
    "causal_judgement": extract_answer_causal_judgement,
    "date_understanding": extract_answer_date_understanding,
    "disambiguation_qa": extract_answer_disambiguation_qa,
    "dyck_languages": extract_answer_dyck_languages,
    "formal_fallacies": extract_answer_formal_fallacies,
    "geometric_shapes": extract_answer_geometric_shapes,
    "hyperbaton": extract_answer_hyperbaton,
    "logical_deduction_five_objects": extract_answer_logical_deduction,
    "logical_deduction_seven_objects": extract_answer_logical_deduction,
    "logical_deduction_three_objects": extract_answer_logical_deduction,
    "movie_recommendation": extract_answer_movie_recommendation,
    "multistep_arithmetic_two": extract_answer_multistep_arithmetic_two,
    "navigate": extract_answer_navigate,
    "object_counting": extract_answer_object_counting,
    "penguins_in_a_table": extract_answer_penguins_in_a_table,
    "reasoning_about_colored_objects": extract_answer_reasoning_about_colored_objects,
    "ruin_names": extract_answer_ruin_names,
    "salient_translation_error_detection": extract_answer_salient_translation_error_detection,
    "snarks": extract_answer_snarks,
    "sports_understanding": extract_answer_sports_understanding,
    "temporal_sequences": extract_answer_temporal_sequences,
    "tracking_shuffled_objects_five_objects": tracking_shuffled_objects,
    "tracking_shuffled_objects_seven_objects": tracking_shuffled_objects,
    "tracking_shuffled_objects_three_objects": tracking_shuffled_objects,
    "web_of_lies": extract_answer_web_of_lies,
    "word_sorting": extract_answer_word_sorting
}

files = glob.glob(f"{prediction_dir}/*.parquet")
acc = []
for prediction_file in files:
    subset = prediction_file.split("|")[1].split(":")[1]
    eval_func = eval_funcs[subset]
    df = pd.read_parquet(prediction_file).to_dict("records")

    num_correct = 0
    for example in df:
        gold = example['gold'][0]
        
        prediction = example['predictions'][0]
        prediction = eval_func(prediction)
        # print(gold, '\t', prediction, example['predictions'][0])
        # print("-" * 20)
        if gold and prediction:
            num_correct += (gold == prediction)

    acc.append(num_correct / len(df) * 100)

print(f"Accuracy: {sum(acc) / len(acc)}")