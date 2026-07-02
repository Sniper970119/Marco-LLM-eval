# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval. We need to use custom tasks for these benchmarks because many of the pre-existing tasks in LightEval are designed for different configurations or base models and thus we must adapt the prompt and metrics to the zero-shot generative case.

Usage:

  # 采样评估（示例，temperature>0）
  lighteval vllm "model_name=HuggingFaceTB/SmolLM3-3B,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" \
      "custom|gsm_plus|0|0,custom|mixeval_hard|0|0" \
      --use-chat-template --output-dir evals/ --custom-tasks tasks.py --save-details

  # pass@1（每题 1 个 greedy 生成）：在 generation_parameters 里设 temperature:0，即 greedy decoding
  lighteval vllm "model_name=...,generation_parameters={max_new_tokens:8192,temperature:0}" \
      "custom|mmlu_it:abstract_algebra|0|0" \
      --use-chat-template --output-dir evals/ --custom-tasks tasks.py --save-details

  # 同时推理多条以加速（连续批处理）：在 model_args 里加 max_num_seqs（默认约 256，显存紧张可调小）
  lighteval vllm "model_name=...,dtype=bfloat16,max_model_length=8192,gpu_memory_utilization=0.9,max_num_seqs=256,generation_parameters={max_new_tokens:4096,temperature:0}" \
      "custom|mmlu_it:abstract_algebra|0|0" \
      --use-chat-template --output-dir evals/ --custom-tasks tasks.py --save-details
"""
from dis import Instruction
from functools import partial
import re
import numpy as np
import json
from typing import Callable
import logging

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
    multilingual_quasi_exact_match_metric,
    multilingual_quasi_f1_score_metric,
)
from lighteval.metrics.metrics import Metrics, MetricCategory
from lighteval.metrics.metrics_sample import (
    JudgeLLM,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm, helm_normalizer
from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.extended.mix_eval.main import (
    flow_judge_for_freeform_template,
    flow_judge_for_multichoice_template,
    mean_dv_5,
    mixeval_freeform_prompt,
    mixeval_multichoice_prompt,
    process_judge_response,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import winogrand_adapter
from lighteval.tasks.multilingual.tasks import TASKS_TABLE as ML_TASKS_TABLE
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.templates.boolq import get_boolq_prompt_function
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro
# from lighteval.utils.utils import remove_reasoning_tags

logger = logging.getLogger(__name__)

import nltk
nltk.download('punkt_tab')

TASKS_TABLE = []
TASKS_TABLE.extend(ML_TASKS_TABLE)

#------------------
# IT MODEL EVALS
#------------------
qa_metrics = [
    loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
    loglikelihood_acc_metric(normalization=LogProbCharNorm()),
]
all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]

def multiple_choice_extractive_match_metric(
    aggregation_function: Callable[[list[float]], float] = max,
    precision: int = 6,
    timeout_seconds: int = 5,
) -> SampleLevelMetric:

    def sample_level_fn(golds: list[str], predictions: list[str], formatted_doc: Doc) -> float:
        def extract_target_from_pred(pred):
            if pred is None:
                return []
            if not isinstance(pred, str):
                pred = str(pred)
            if "\n" in pred:
                pred = pred.split("\n")[0]
            pred_stripped = pred.strip()
            # 1) JSON/dict: {"answer": "A"}
            try:
                parsed = eval(pred_stripped)
                if isinstance(parsed, dict) and "answer" in parsed:
                    return [str(parsed["answer"]).strip().upper()]
            except Exception:
                pass
            # 2) "Answer: A" / "answer: A" / "ANSWER: B"
            m = re.search(r'answer\s*:\s*["\']?([A-Da-d])["\']?', pred_stripped, re.I)
            if m:
                return [m.group(1).upper()]
            # 3) "A." / "A" / 纯字母
            for letter in LETTER_INDICES:
                if (
                    pred_stripped.startswith(f"{letter}.")
                    or f"{letter}." in pred_stripped
                    or pred_stripped.strip() == letter
                ):
                    return [letter]
            return []

        extracted_golds = [[gold] for gold in golds]
        extracted_predictions = [
            extract_target_from_pred(pred) for pred in predictions
        ]

         # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            logger.warning(f"We did not manage to extract a gold in the correct format. Gold: {golds}")

        if all(len(p) == 0 for p in extracted_predictions):
            logger.warning(
                f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Pred: {predictions}"
            )
        
        return aggregation_function(
            [
                (
                    1.0
                    if any(
                        compare_gold_target(gold, pred, precision, timeout_seconds=timeout_seconds)
                        for gold in extracted_golds
                    )
                    else 0.0
                )
                for pred in extracted_predictions
            ]
        )
    
    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

multiple_choice_metric = multiple_choice_extractive_match_metric()

# ---------- 统一 0-shot + \\boxed{} 测评 ----------
BOXED_INSTRUCTION = "Please reason step by step and put your answer within \\boxed{}."


def extract_boxed(text: str) -> str:
    """从模型输出中提取 \\boxed{...} 的内容，取最后一个 boxed（通常为最终答案）。支持嵌套花括号。"""
    if not text or not isinstance(text, str):
        return ""
    start_markers = ["\\boxed{", "\\boxed {"]
    for start in start_markers:
        idx = text.rfind(start)  # 最后一个
        if idx == -1:
            continue
        begin = idx + len(start)
        depth = 1
        i = begin
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[begin : i - 1].strip()
    return ""


def extract_after_hashes(text: str) -> str:
    """从模型输出中提取最后一个 ### 或 #### 之后的文本（常用于最终答案）。例如 '...\\n#### 2' -> '2'。"""
    if not text or not isinstance(text, str):
        return ""
    # 优先匹配最后一个 ####，否则最后一个 ###
    for marker in ["####", "###"]:
        idx = text.rfind(marker)
        if idx == -1:
            continue
        rest = text[idx + len(marker) :].strip()
        # 取第一行或整段（若只有一行），去掉首尾空白
        if "\n" in rest:
            rest = rest.split("\n")[0].strip()
        if rest:
            return rest
    return ""


def extract_after_answer_keyword(text) -> str:
    """从模型输出中提取 \"Answer:\" 后的整段文本（不区分大小写）。例如 '...Answer: B. Change the ratio of...' -> 'B. Change the ratio of...'。
    用于判分时若标准答案（选项字母或选项原文）包含在解析出的文本中则判对。"""
    if not text:
        return ""
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    s = (text or "").strip()
    if not isinstance(s, str):
        return ""
    m = re.search(r"Answer\s*:\s*(.+)", s, re.I | re.DOTALL)
    if not m:
        return ""
    rest = m.group(1).strip()
    # 去掉末尾的 markdown 粗体等
    rest = re.sub(r"\*+$", "", rest).strip()
    if "\n" in rest:
        rest = rest.split("\n")[0].strip()
    return rest if rest else ""


def extract_answer_fallback(text) -> str:
    """当没有 \\boxed{} 时，从模型输出中尝试解析答案：答案：A / Answer: A / A. / A) 等。"""
    if not text:
        return ""
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    s = (text or "").strip()
    if not isinstance(s, str):
        return ""
    # 答案：A / 答案：A. / 答案：A. xxx（支持 A–J 多选项）
    m = re.search(r"答案\s*[：:]\s*([A-Ja-j])(?:\.|\.\s|\)|,|\s|$)", s)
    if m:
        return m.group(1).upper()
    # Answer: A / Answer: A.
    m = re.search(r"Answer\s*:\s*([A-Ja-j])(?:\.|\)|,|\s|$)", s, re.I)
    if m:
        return m.group(1).upper()
    # 行首或句中 A. / B. / … 且较靠后（视为最终答案）
    for letter in LETTER_INDICES[::-1]:  # 从后往前优先匹配 J,I,...,A
        idx = s.rfind(f"{letter}.")
        if idx != -1 and (idx + 2 >= len(s) or s[idx + 2] in " \n\t"):
            return letter
    for letter in LETTER_INDICES[::-1]:
        if re.search(rf"\b{letter}\s*[.)]\s*$", s) or re.search(rf"\b{letter}\s*[.)]\s*\n", s):
            return letter
    return ""


def _boxed_compare(gold: str, pred: str, use_math_eval: bool) -> bool:
    """比较 gold 与 pred（均为从 boxed 中解析出的字符串）。不区分大小写；若 use_math_eval 则用 math_verify。"""
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    if use_math_eval:
        try:
            from math_verify import parse, verify
            parsed_gold = parse(gold)
            parsed_pred = parse(pred)
            if parsed_gold and parsed_pred:
                return verify(parsed_gold, parsed_pred)
        except Exception:
            pass
    return pred.lower() == gold.lower()


def boxed_match_metric(use_math_eval: bool = False) -> SampleLevelMetric:
    """统一 metric：从预测中解析 \\boxed{} 内容，与 gold 比较（不区分大小写；数学题用 math_verify）。
    对于有选项的 MCQ（formatted_doc.choices 长度>1）：若 boxed 内容与正确选项字母（A/B/C/…）或与正确选项文本匹配，均判对。"""
    def sample_level_fn(golds: list[str], predictions: list[str], formatted_doc: Doc) -> float:
        extracted_golds = [[g.strip() for g in golds]]
        # 先尝试 \\boxed{}，再尝试 ###/#### 后文本，再尝试 Answer: 后全文，最后用「答案：A」/ Answer: A 等回退解析
        # 仅当来自 boxed 或 ###/#### 时允许「可接受答案包含在解析结果中」；Answer: / 回退解析只做精确匹配，避免单词中含 b 等被误判对
        extracted_preds = []
        for p in predictions:
            raw = extract_boxed(p)
            from_boxed_or_hashes = bool(raw)
            if not raw:
                raw = extract_after_hashes(p)
                from_boxed_or_hashes = bool(raw)
            if not raw:
                raw = extract_after_answer_keyword(p)
            if not raw:
                raw = extract_answer_fallback(p)
            extracted_preds.append((raw, from_boxed_or_hashes))
        if any(len(g) == 0 for g in extracted_golds):
            logger.warning(f"Empty gold. Gold: {golds}")
        scores = []
        for gold_list, (pred, from_boxed_or_hashes) in zip(extracted_golds, extracted_preds):
            if not gold_list:
                scores.append(0.0)
                continue
            gold = gold_list[0]
            if not pred:
                logger.debug("Could not extract answer from prediction (no \\boxed{}, ###/####, Answer:, nor 答案：A/Answer: A). Gold: %s, Pred: %s", gold, predictions)
                scores.append(0.0)
                continue
            if use_math_eval:
                scores.append(1.0 if _boxed_compare(gold, pred, use_math_eval) else 0.0)
            else:
                # 选择题：gold 为正确选项原文，可接受答案为选项字母或选项原文（不区分大小写）
                # boxed/###/#### 解析：允许「可接受答案包含在解析结果中」；Answer:/回退解析：仅精确匹配（避免单词中含 b 等误判对）
                acceptable = [gold.lower()]
                choices = getattr(formatted_doc, "choices", None)
                gold_index = getattr(formatted_doc, "gold_index", None)
                if isinstance(choices, (list, tuple)) and len(choices) > 1 and gold_index is not None and 0 <= gold_index < len(LETTER_INDICES):
                    acceptable.append(LETTER_INDICES[gold_index].lower())
                specific = getattr(formatted_doc, "specific", None) or {}
                opt_texts = specific.get("option_texts")
                opt_idx = specific.get("gold_index")
                if isinstance(opt_texts, (list, tuple)) and opt_idx is not None and 0 <= opt_idx < len(opt_texts):
                    acceptable.append((opt_texts[opt_idx] or "").strip().lower())
                pred_norm = (pred or "").strip().lower()
                if from_boxed_or_hashes:
                    match = pred_norm in acceptable or any(acc in pred_norm for acc in acceptable if acc)
                else:
                    match = pred_norm in acceptable
                scores.append(1.0 if match else 0.0)
        return max(scores) if scores else 0.0

    return SampleLevelMetric(
        metric_name="boxed_math" if use_math_eval else "boxed_match",
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


# 选择题/非数学：只解析 boxed，不区分大小写比较
unified_boxed_metric = boxed_match_metric(use_math_eval=False)
# 数学题：解析 boxed 后用 math_verify 比较（若未安装则回退为字符串比较）
unified_boxed_math_metric = boxed_match_metric(use_math_eval=True)

latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=max,
)

# custom|boolq_it|0|0
# custom|commonsenseqa_it|0|0
# custom|openbookqa_it|0|0
# custom|piqa_it|0|0
# custom|siqa_it|0|0
def get_arc_it_prompt(line, task_name):
    query = f"Question: {line['question']}"
    option_texts = line["choices"]["text"]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = LETTER_INDICES[int(line["answerKey"]) - 1] if line["answerKey"].isdigit() else line["answerKey"].upper()
    gold_index = int(line["answerKey"]) - 1 if line["answerKey"].isdigit() else LETTER_INDICES.index(gold_letter)
    return Doc(
        task_name=task_name,
        query=query + "\n" + BOXED_INSTRUCTION,
        choices=option_texts,
        gold_index=gold_index,
    )

# ARC tasks
arc_tasks = [
    LightevalTaskConfig(
        name=f"arc_it:{subset.lower()}",
        prompt_function=get_arc_it_prompt,
        suite=("custom",),
        hf_repo="allenai/ai2_arc",
        hf_subset=f"ARC-{subset}",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subset in ["Easy", "Challenge"]
]
TASKS_TABLE.extend(arc_tasks)

# multilingual ARC challenge tasks
mlmm_arc_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_arc_it_{lang.value}",
        prompt_function=get_arc_it_prompt,
        suite=("custom",),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(lang.value),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for lang in [
        Language.ARABIC,
        Language.BENGALI,
        Language.GERMAN,
        Language.SPANISH,
        Language.FRENCH,
        Language.HUNGARIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.DUTCH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
]
TASKS_TABLE.extend(mlmm_arc_tasks)

def get_boolq_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    option_texts = ['Yes', 'No']
    query = f"{line['passage']}\nQuestion: {line['question']}"
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_index = 0 if line['answer'] else 1
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# BoolQ task
boolq_task = LightevalTaskConfig(
    name="boolq_it",
    prompt_function=get_boolq_it_prompt,
    suite=("custom",),
    hf_repo="google/boolq",
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split=None,
    generation_size=8192,
    stop_sequence=None,
    metric=[unified_boxed_metric],
)
TASKS_TABLE.append(boolq_task)

# HellaSwag tasks
hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_{formulation.name.lower()}",
        suite=["custom"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.ENGLISH,
            adapter=lambda line: {
                "activity_label": line["activity_label"],
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        evaluation_splits=["validation"],
        hf_avail_splits=["train", "validation"],
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(hellaswag_tasks)

mlmm_hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag_{lang.value}_{formulation.name.lower()}",
        suite=["custom"],
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for lang in [
        Language.ARABIC,
        Language.BENGALI,
        Language.GERMAN,
        Language.SPANISH,
        Language.FRENCH,
        Language.HUNGARIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.DUTCH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]
TASKS_TABLE.extend(mlmm_hellaswag_tasks)

def get_commonsense_qa_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question']}"
    labels = line['choices']['label']
    option_texts = line['choices']['text']
    for label, choice in zip(labels, option_texts):
        query += f"\n{label}. {choice}"
    gold_key = line['answerKey'].upper() if isinstance(line['answerKey'], str) else line['answerKey']
    gold_index = labels.index(gold_key) if gold_key in labels else (LETTER_INDICES.index(gold_key) if gold_key in LETTER_INDICES else 0)
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# CommonsenseQA tasks
commonsense_qa_tasks = [
    LightevalTaskConfig(
        name=f"commonsenseqa_it",
        prompt_function=get_commonsense_qa_it_prompt,
        suite=("custom",),
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_revision="94630fe30dad47192a8546eb75f094926d47e155",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
]
TASKS_TABLE.extend(commonsense_qa_tasks)

def get_openbook_qa_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question_stem']}"
    labels = line['choices']['label']
    option_texts = line['choices']['text']
    for label, choice in zip(labels, option_texts):
        query += f"\n{label}. {choice}"
    gold_key = line['answerKey'].upper() if isinstance(line['answerKey'], str) else line['answerKey']
    gold_index = labels.index(gold_key) if gold_key in labels else (LETTER_INDICES.index(gold_key) if gold_key in LETTER_INDICES else 0)
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# OpenBookQA tasks
openbook_qa_tasks = [
    LightevalTaskConfig(
        name="openbookqa_it",
        prompt_function=get_openbook_qa_it_prompt,
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
]
TASKS_TABLE.extend(openbook_qa_tasks)

# Winogrande tasks
winogrande_tasks = [
    LightevalTaskConfig(
        name=f"winogrande_{formulation.name.lower()}",
        suite=("custom",),
        prompt_function=get_continuation_prompt_function(
            Language.ENGLISH,
            partial(winogrand_adapter, Language.ENGLISH),
            formulation=formulation,
        ),
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        trust_dataset=True,
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        metric=qa_metrics,
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(winogrande_tasks)

def get_piqa_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    option_texts = [line["sol1"], line["sol2"]]
    query = f"Question: {line['goal']}"
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_index = int(line["label"])
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# PIQA tasks
piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_it",
        prompt_function=get_piqa_it_prompt,
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        hf_subset="plain_text",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
]
TASKS_TABLE.extend(piqa_tasks)

def get_siqa_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    option_texts = [line["answerA"], line["answerB"], line["answerC"]]
    query = f"{line['context']}\nQuestion: {line['question']}"
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_index = int(line["label"]) - 1
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# SIQA tasks
siqa_tasks = [
    LightevalTaskConfig(
        name=f"siqa_it",
        prompt_function=get_siqa_it_prompt,
        suite=["custom"],
        hf_repo="allenai/social_i_qa",
        hf_revision="53620e5841fb12b08e082485797e7021d3684ea2",
        hf_subset="default",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
]
TASKS_TABLE.extend(siqa_tasks)

# MMLU tasks
# fmt: off
MMLU_DEBUG = False  # True=只跑 2 个子集快速 debug，False=跑全部 57 个子集
MMLU_SUBSETS_FULL = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
    'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]
MMLU_SUBSETS = ["abstract_algebra", "anatomy"] if MMLU_DEBUG else MMLU_SUBSETS_FULL
def get_mmlu_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question']}"
    option_texts = line["choices"]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_index = int(line["answer"])
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_it:{subset}",
        prompt_function=get_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="cais/mmlu",
        hf_subset=subset,
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subset in MMLU_SUBSETS
]
TASKS_TABLE.extend(mmlu_tasks)

mmlu_redux_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_redux_it:{subset}",
        prompt_function=get_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_subset=subset,
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subset in MMLU_SUBSETS
]
TASKS_TABLE.extend(mmlu_redux_tasks)

def get_mmlu_pro_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question']}"
    if "options" not in line:
        line["options"] = [line[f"option_{i}"] for i in range(10)]
    for i in range(len(line["options"])):
        query += f"\n{LETTER_INDICES[i]}. {line['options'][i]}"
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[line['answer']],
        gold_index=0,
    )

# MMLU Pro IT tasks
mmlu_pro_it_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_it",
        prompt_function=get_mmlu_pro_it_prompt,
        suite=("custom",),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
        version=0,
    )
]
TASKS_TABLE.extend(mmlu_pro_it_tasks)

# MMLU Pro X IT tasks
mmlu_pro_x_it_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_x_it_{language.value}",
        prompt_function=get_mmlu_pro_it_prompt,
        suite=("custom",),
        hf_repo="li-lab/MMLU-ProX",
        hf_subset=standardize_tag(language.value),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
        version=0,
    )
    for language in [
        Language.CHINESE,
        Language.JAPANESE,
        Language.KOREAN,
        Language.FRENCH,
        Language.GERMAN,
        Language.SPANISH,
        Language.PORTUGUESE,
        Language.ARABIC,
        Language.THAI,
        Language.BENGALI,
        Language.CZECH,
        Language.HUNGARIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.RUSSIAN,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
        Language.ENGLISH,
    ]
]
TASKS_TABLE.extend(mmlu_pro_x_it_tasks)

mmlu_pro_x_lite_it_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_x_lite_it_{language.value}",
        prompt_function=get_mmlu_pro_it_prompt,
        suite=("custom",),
        hf_repo="li-lab/MMLU-ProX-Lite",
        hf_subset=standardize_tag(language.value),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
        version=0,
    )
    for language in [
        Language.CHINESE,
        Language.JAPANESE,
        Language.KOREAN,
        Language.FRENCH,
        Language.GERMAN,
        Language.SPANISH,
        Language.PORTUGUESE,
        Language.ARABIC,
        Language.THAI,
        Language.BENGALI,
        Language.CZECH,
        Language.HUNGARIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.RUSSIAN,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
        Language.ENGLISH,
    ]
]
TASKS_TABLE.extend(mmlu_pro_x_lite_it_tasks)

def get_gpqa_cot_prompt_function(line, task_name):
    def preprocess(text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    GPQA_QUERY_TEMPLATE = """Question: {Question}
Choices:
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}
"""
    choices = [
        preprocess(line["Incorrect Answer 1"]),
        preprocess(line["Incorrect Answer 2"]),
        preprocess(line["Incorrect Answer 3"]),
        preprocess(line["Correct Answer"]),
    ]
    import random
    random.shuffle(choices)
    correct_answer_index = choices.index(preprocess(line["Correct Answer"]))
    out_doc = {
        "choice1": choices[0],
        "choice2": choices[1],
        "choice3": choices[2],
        "choice4": choices[3],
    }
    query = GPQA_QUERY_TEMPLATE.format(Question=line["Question"], **out_doc)
    target = chr(65 + correct_answer_index)
    return Doc(
        task_name=task_name,
        query=query + BOXED_INSTRUCTION,
        choices=[target],
        gold_index=0,
    )

# GPQA tasks
gpqa_cot_tasks = [
    LightevalTaskConfig(
        name=f"gpqa_cot",
        prompt_function=get_gpqa_cot_prompt_function,
        suite=("custom",),
        hf_repo="Idavidrein/gpqa",
        hf_subset="gpqa_main",
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split=None,
        generation_size=8192,
        metric=[unified_boxed_metric],
        stop_sequence=["Question:"],
        version=0,
    )
]
TASKS_TABLE.extend(gpqa_cot_tasks)

# multilingual GPQA
# use English gpqa few-shot.
mgpqa_cot_tasks = [
    LightevalTaskConfig(
        name=f"gpqa_cot_{language}",
        prompt_function=get_gpqa_cot_prompt_function,
        suite=("custom",),
        hf_repo="LLaMAX/BenchMAX_Science",
        hf_subset=language,
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split=None,
        generation_size=8192,
        metric=[unified_boxed_metric],
        stop_sequence=["Question:"],
        version=0,
    )
    for language in [
        'ar',
        'bn',
        'zh',
        'cs',
        'en',
        'fr',
        'de',
        'hu',
        'ja',
        'ko',
        'es',
        'th',
        'ru',
        'vi',
    ]
]
TASKS_TABLE.extend(mgpqa_cot_tasks)

def get_super_gpqa_cot_prompt_function(line, task_name):
    ids2choice = {
        0: 'A)',
        1: 'B)',
        2: 'C)',
        3: 'D)',
        4: 'E)',
        5: 'F)',
        6: 'G)',
        7: 'H)',
        8: 'I)',
        9: 'J)',
    }
    GPQA_QUERY_TEMPLATE = """Question: {Question}
"""
    query = GPQA_QUERY_TEMPLATE.format(Question=line["question"])
    for i in range(len(line['options'])):
        query += ids2choice[i] + " " + line['options'][i] + "\n"
    query += "\n" + BOXED_INSTRUCTION
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["answer_letter"]],
        gold_index=0,
    )

# super gpqa tasks
supergpqa_cot_tasks = [
    LightevalTaskConfig(
        name=f"supergpqa_cot",
        prompt_function=get_super_gpqa_cot_prompt_function,
        suite=("custom",),
        hf_repo="m-a-p/SuperGPQA",
        hf_subset="default",
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split=None,
        generation_size=8192,
        metric=[unified_boxed_metric],
        stop_sequence=["Question:"],
        version=0,
    )
]
TASKS_TABLE.extend(supergpqa_cot_tasks)

def get_belebele_it_prompt(line, task_name):
    """0-shot: passage + question + 4 choices + BOXED_INSTRUCTION. 判分时 A/B/C/D 与选项原文均算对。"""
    query = f"{line['flores_passage']}\n\nQuestion: {line['question']}\n"
    option_texts = [line[f'mc_answer{i}'] for i in range(1, 5)]
    for i, text in enumerate(option_texts):
        query += f"{LETTER_INDICES[i]}. {text}\n"
    query += "\n" + BOXED_INSTRUCTION
    gold_index = int(line["correct_answer_num"]) - 1
    return Doc(
        task_name=task_name,
        query=query,
        choices=option_texts,
        gold_index=gold_index,
    )

# BELEBELE tasks
belebele_tasks = [
    LightevalTaskConfig(
        name=f"belebele_it_{language}",
        prompt_function=get_belebele_it_prompt,
        suite=("custom",),
        hf_repo="facebook/belebele",
        hf_subset=language,
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for language in [
        "ita_Latn",
        "pol_Latn",
        "ces_Latn",
        "fra_Latn",
        "por_Latn",
        "jpn_Jpan",
        "ukr_Cyrl",
        "zho_Hans",
        "hun_Latn",
        "kor_Hang",
        "ron_Latn",
        "urd_Arab",
        "ben_Beng",
        "deu_Latn",
        "rus_Cyrl",
        "zsm_Latn",
        "arb_Arab",
        "ell_Grek",
        "spa_Latn",
        "eng_Latn",
        "tha_Thai",
        "vie_Latn",
        "ind_Latn",
        "nld_Latn",
        "heb_Hebr",
        "kaz_Cyrl",
        "tur_Latn",
    ]
]
TASKS_TABLE.extend(belebele_tasks)

# SIB-200 tasks
sib200_tasks = [
    LightevalTaskConfig(
        name=f"sib200_{language}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            # iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language if language not in ['heb_Hebr', 'ben_Beng', 'kaz_Cyrl', 'kor_Hang'] else 'eng_Latn').to_alpha3()],
            iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get('eng_Latn').to_alpha3()],
            lambda line: {
                "question": "What label best describes this news article?",
                "context": line["text"],
                "choices": ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"],
                "gold_idx": ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"].index(line["category"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="Davlan/sib200",
        hf_subset=language,
        evaluation_splits=("train", "validation", "test",),
        hf_avail_splits=["test"],
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
    for language in [
        "ita_Latn",
        "pol_Latn",
        "ces_Latn",
        "fra_Latn",
        "por_Latn",
        "jpn_Jpan",
        "ukr_Cyrl",
        "zho_Hans",
        "hun_Latn",
        "kor_Hang",
        "ron_Latn",
        "urd_Arab",
        "ben_Beng",
        "deu_Latn",
        "rus_Cyrl",
        "zsm_Latn",
        "arb_Arab",
        "ell_Grek",
        "spa_Latn",
        "eng_Latn",
        "tha_Thai",
        "vie_Latn",
        "ind_Latn",
        "nld_Latn",
        "heb_Hebr",
        "kaz_Cyrl",
        "tur_Latn",
    ]
]
TASKS_TABLE.extend(sib200_tasks)

mcq_step_by_step_prompts = {
    Language.ARABIC: "يرجى شرح الحل خطوة بخطوة، ووضع إجابتك النهائية داخل المربع \\boxed{}",
    Language.BENGALI: "অনুগ্রহ করে ধাপে ধাপে যুক্তি দিন এবং আপনার উত্তর \\boxed{} এর মধ্যে লিখুন।",
    Language.CHINESE: "请逐步展开推理，并将最终答案放在\\boxed{}内。",
    Language.DUTCH: "Leg je redenering stap voor stap uit en plaats je eindantwoord tussen \\boxed{}.",
    Language.FRENCH: "Veuillez raisonner étape par étape et inscrire votre réponse finale dans \\boxed{}.",
    Language.GERMAN: "Bitte begründen Sie Ihre Antwort Schritt für Schritt und geben Sie Ihr Endergebnis innerhalb von \\boxed{} an.",
    Language.GREEK: "Παρακαλώ συλλογιστείτε βήμα προς βήμα και βάλτε την τελική σας απάντηση εντός \\boxed{}.",
    Language.HEBREW: "אנא נמק שלב אחר שלב, וסמן את תשובתך הסופית בתוך \\boxed{}",
    Language.HUNGARIAN: "Kérem, indokolja lépésről lépésre, és a végső válaszát helyezze \\boxed{} keretbe.",
    Language.INDONESIAN: "Mohon jelaskan langkah demi langkah dan masukkan jawaban Anda di dalam \\boxed{}.",
    Language.ITALIAN: "Si prega di ragionare passo dopo passo, e di racchiudere la risposta finale tra \\boxed{}.",
    Language.JAPANESE: "段階的に説明し、最終的な回答は\\boxed{}の中に入れてください。",
    Language.KAZAKH: "Өтінемін, қадамдап негіздеңіз және соңғы жауабыңызды \\boxed{} ішіне орналастырыңыз.",
    Language.KOREAN: "단계별로 추론해 주시고, 최종 답변을 \\boxed{} 안에 넣어 주세요.",
    Language.MALAY: "Sila berikan alasan langkah demi langkah, dan letakkan jawapan akhir anda di dalam \\boxed{}.",
    Language.POLISH: "Proszę uzasadnić krok po kroku i umieścić swoją ostateczną odpowiedź w \\boxed{}.",
    Language.PORTUGUESE: "Por favor, raciocine passo a passo, e coloque sua resposta final dentro de \\boxed{}.",
    Language.RUSSIAN: "Пожалуйста, обоснуйте шаг за шагом и поместите ваш окончательный ответ в \\boxed{}.",
    Language.SPANISH: "Por favor, razone paso a paso, y ponga su respuesta final dentro de \\boxed{}.",
    Language.TURKISH: "Lütfen adım adım gerekçelerinizi açıklayın ve cevabınızı \\boxed{} içine yazın.",
    Language.UKRAINIAN: "Будь ласка, обґрунтуйте крок за кроком і помістіть свою остаточну відповідь у \\boxed{}.",
    Language.URDU: "براہ کرم قدم بہ قدم دلیل دیں، اور اپنا حتمی جواب \\boxed{} کے اندر رکھیں۔",
    Language.VIETNAMESE: "Vui lòng lập luận từng bước, và đặt đáp án cuối cùng của bạn vào trong \\boxed{}.",
    Language.ENGLISH: "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
}

def get_include_it_prompt(line, task_name, language):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question']}"
    option_texts = [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_index = line["answer"]  # 0-based index
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# INCLUDE tasks
include_it_tasks = [
    LightevalTaskConfig(
        name=f"include_it_{formation.lower()}_{language.value}",
        prompt_function=partial(get_include_it_prompt, language=language if formation == 'Native' else Language.ENGLISH),
        suite=("custom",),
        hf_repo="CohereLabs/include-base-44",
        hf_subset=str(language).split(".")[1].capitalize(),
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for language in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CHINESE,
        Language.DUTCH,
        Language.FRENCH,
        Language.GERMAN,
        Language.GREEK,
        Language.HEBREW,
        Language.HUNGARIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KAZAKH,
        Language.KOREAN,
        Language.MALAY,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SPANISH,
        Language.TURKISH,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
    ]
    for formation in ['Native', 'English']
]
TASKS_TABLE.extend(include_it_tasks)

def get_global_mmlu_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    option_texts = [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]
    query = f"Question: {line['question']}"
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = line["answer"].upper() if isinstance(line["answer"], str) else LETTER_INDICES[int(line["answer"]) - 1]
    gold_index = LETTER_INDICES.index(gold_letter) if gold_letter in LETTER_INDICES else 0
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# global_mmlu tasks
global_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"global_mmlu_it_{language.value}",
        prompt_function=get_global_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for language in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CHINESE,
        Language.CZECH,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.GREEK,
        Language.FRENCH,
        Language.HEBREW,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.MALAY,
        Language.NEPALI,
        Language.DUTCH,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.THAI,
        Language.TURKISH,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
    ]
]
TASKS_TABLE.extend(global_mmlu_tasks)

# -------------------- Language Specific MMLU tasks --------------------
def get_cmmlu_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    option_texts = [line["A"], line["B"], line["C"], line["D"]]
    query = f"Question: {line['question']}"
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = line["answer"].upper() if isinstance(line["answer"], str) else LETTER_INDICES[int(line["answer"]) - 1]
    gold_index = LETTER_INDICES.index(gold_letter) if gold_letter in LETTER_INDICES else 0
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

cmmlu_tasks = [
    LightevalTaskConfig(
        name=f"zho_mmlu_it:{subset}",
        prompt_function=get_cmmlu_it_prompt,
        suite=("custom",),
        hf_repo="lmlmcat/cmmlu",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subset in [
        "agronomy",
        "anatomy",
        "ancient_chinese",
        "arts",
        "astronomy",
        "business_ethics",
        "chinese_civil_service_exam",
        "chinese_driving_rule",
        "chinese_food_culture",
        "chinese_foreign_policy",
        "chinese_history",
        "chinese_literature",
        "chinese_teacher_qualification",
        "clinical_knowledge",
        "college_actuarial_science",
        "college_education",
        "college_engineering_hydrology",
        "college_law",
        "college_mathematics",
        "college_medical_statistics",
        "college_medicine",
        "computer_science",
        "computer_security",
        "conceptual_physics",
        "construction_project_management",
        "economics",
        "education",
        "electrical_engineering",
        "elementary_chinese",
        "elementary_commonsense",
        "elementary_information_and_technology",
        "elementary_mathematics",
        "ethnology",
        "food_science",
        "genetics",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_geography",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "human_sexuality",
        "international_law",
        "journalism",
        "jurisprudence",
        "legal_and_moral_basis",
        "logical",
        "machine_learning",
        "management",
        "marketing",
        "marxist_theory",
        "modern_chinese",
        "nutrition",
        "philosophy",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_study",
        "sociology",
        "sports_science",
        "traditional_chinese_medicine",
        "virology",
        "world_history",
        "world_religions",
    ]
]
TASKS_TABLE.extend(cmmlu_tasks)

# arabic mmlu tasks
def get_arabic_mmlu_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question']}"
    option_texts = [str(o) for o in [line[f"Option {i}"] for i in range(1, 6)] if o]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_key = line["Answer Key"]
    gold_index = LETTER_INDICES.index(gold_key.upper()) if isinstance(gold_key, str) and gold_key.upper() in LETTER_INDICES else (int(gold_key) - 1 if str(gold_key).isdigit() else 0)
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

arabic_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"ara_mmlu_it:{subset}",
        prompt_function=get_arabic_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="MBZUAI/ArabicMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        hf_avail_splits=["dev"],
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subset in [
        "Islamic Studies",
        "Islamic Studies (Middle School)",
        "Islamic Studies (Primary School)",
        "Islamic Studies (High School)",
        "Driving Test",
        "Natural Science (Middle School)",
        "Natural Science (Primary School)",
        "History (Middle School)",
        "History (Primary School)",
        "History (High School)",
        "General Knowledge",
        "General Knowledge (Middle School)",
        "General Knowledge (Primary School)",
        "Law (Professional)",
        "Physics (High School)",
        "Social Science (Middle School)",
        "Social Science (Primary School)",
        "Management (University)",
        "Arabic Language (Middle School)",
        "Arabic Language (Primary School)",
        "Arabic Language (High School)",
        "Political Science (University)",
        "Philosophy (High School)",
        "Accounting (University)",
        "Computer Science (Middle School)",
        "Computer Science (Primary School)",
        "Computer Science (High School)",
        "Computer Science (University)",
        "Geography (Middle School)",
        "Geography (Primary School)",
        "Geography (High School)",
        "Math (Primary School)",
        "Biology (High School)",
        "Economics (Middle School)",
        "Economics (High School)",
        "Economics (University)",
        "Arabic Language (General)",
        "Arabic Language (Grammar)",
        "Civics (Middle School)",
        "Civics (High School)",
    ]
]
TASKS_TABLE.extend(arabic_mmlu_tasks)

# turkish mmlu tasks
def get_turkish_mmlu_it_prompt(line, task_name, language):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question']}"
    option_texts = line["choices"]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = line["answer"].upper() if isinstance(line["answer"], str) else LETTER_INDICES[int(line["answer"])]
    gold_index = LETTER_INDICES.index(gold_letter) if gold_letter in LETTER_INDICES else int(line["answer"])
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

turkish_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"tur_mmlu_it_{formation.lower()}:{subset}",
        prompt_function=partial(get_turkish_mmlu_it_prompt, language=Language.TURKISH if formation == 'Native' else Language.ENGLISH),
        suite=("custom",),
        hf_repo="AYueksel/TurkishMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subset in [
        "Biology",
        "Chemistry",
        "Geography",
        "History",
        "Mathematics",
        "Philosophy",
        "Physics",
        "Religion_and_Ethics",
        "Turkish_Language_and_Literature",
    ]
    for formation in ['Native', 'English']
]
TASKS_TABLE.extend(turkish_mmlu_tasks)

# KazMMLU tasks
def get_kaz_mmlu_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['Question']}"
    option_texts = [line["Option A"], line["Option B"], line["Option C"], line["Option D"], line["Option E"]]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_key = line["Answer Key"].upper() if isinstance(line["Answer Key"], str) else line["Answer Key"]
    gold_index = LETTER_INDICES.index(gold_key) if gold_key in LETTER_INDICES else 0
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

kaz_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"kaz_mmlu_it:{subject}",
        prompt_function=get_kaz_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="MBZUAI/KazMMLU",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subject in [
        "Accounting and Auditing (Professional & University in rus)",
        "Biology (High School in kaz)",
        "Accounting and Auditing (Professional & University in rus)",
        "Biology (High School in kaz)",
        "Biology (High School in rus)",
        "Biology (Professional & University in rus)",
        "Chemistry (High School in kaz)",
        "Chemistry (High School in rus)",
        "Culture and Art (Professional & University in rus)",
        "Economics and Entrepreneurship (Professional in rus)",
        "Education and Training (Professional & University in rus)",
        "Finance (Professional & University in rus)",
        "General Education Disciplines (Professional & University in rus)",
        "Geography (High School in kaz)",
        "Geography (High School in rus)",
        "Informatics (High School in kaz)",
        "Informatics (High School in rus)",
        "Jurisprudence (Professional & University in rus)",
        "Kazakh History (High School in kaz)",
        "Kazakh History (High School in rus)",
        "Kazakh Language (High School in kaz)",
        "Kazakh Literature (High School in kaz)",
        "Law (High School in kaz)",
        "Law (High School in rus)",
        "Management and Marketing (Professional & University in rus)",
        "Math (High School in kaz)",
        "Math (High School in rus)",
        "Math Literacy (High School in rus)",
        "Medicine (Professional & University in rus)",
        "Philosophy and Psychology (Professional & University in rus)",
        "Physics (High School in kaz)",
        "Physics (High School in rus)",
        "Reading Literacy (High School in kaz)",
        "Reading Literacy (High School in rus)",
        "Russian Language (High School in rus)",
        "Russian Literature (High School in rus)",
        "Social Science (Professional & University in rus)",
        "World History (High School in kaz)",
        "World History (High School in rus)",
    ]
]
TASKS_TABLE.extend(kaz_mmlu_tasks)

def get_indo_career_it_prompt(line, task_name, subject, language):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['Question']}"
    option_texts = [line["Option A"], line["Option B"], line["Option C"], line["Option D"], line["Option E"]]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = line["Answer Key"].upper() if isinstance(line["Answer Key"], str) else LETTER_INDICES[int(line["Answer Key"]) - 1]
    gold_index = LETTER_INDICES.index(gold_letter) if gold_letter in LETTER_INDICES else 0
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# IndoCareer tasks
indo_career_tasks = [
    LightevalTaskConfig(
        name=f"indo_career_it_{formation.lower()}:{subject}",
        prompt_function=partial(get_indo_career_it_prompt, subject=subject, language=Language.INDONESIAN if formation == "Native" else Language.ENGLISH),
        suite=("custom",),
        hf_repo="indolem/IndoCareer",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for subject in [
        "Advocate",
        "Broadcasting",
        "Certified Financial Planner",
        "Certified Indonesian Tax Accountant",
        "Certified Professional Management Accountant",
        "Certified Public Accountant",
        "Clinical Psychology",
        "Culinary Art",
        "Fashion Design",
        "Graphic Design",
        "Hospitality",
        "Life Insurance",
        "Medical Doctor",
        "Midwife",
        "Nurse",
        "Office Administration",
        "Pharmacist",
        "Police",
        "Risk Management",
        "Sharia Life Insurance",
        "Teacher Competency Test",
        "Tourism",
    ]
    for formation in ['Native', 'English']
]
TASKS_TABLE.extend(indo_career_tasks)

def get_indo_culture_it_prompt(line, task_name, language):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['context']}"
    option_texts = [option[3:] for option in eval(line["options"])]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = line["answer"].upper() if isinstance(line["answer"], str) else LETTER_INDICES[int(line["answer"])]
    gold_index = LETTER_INDICES.index(gold_letter) if gold_letter in LETTER_INDICES else 0
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

# IndoCulture tasks
indo_culture_tasks = [
    LightevalTaskConfig(
        name=f"indo_culture_it_{formation.lower()}",
        prompt_function=partial(get_indo_culture_it_prompt, language=Language.INDONESIAN if formation == "Native" else Language.ENGLISH),
        suite=("custom",),
        hf_repo="indolem/IndoCulture",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for formation in ['Native', 'English']
]
TASKS_TABLE.extend(indo_culture_tasks)

# IndoMMLU tasks
def get_indo_mmlu_it_prompt(line, task_name, language):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['question']}"
    option_texts = [option[3:] for option in eval(line["options"])]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = line["answer"].upper() if isinstance(line["answer"], str) else LETTER_INDICES[int(line["answer"])]
    gold_index = LETTER_INDICES.index(gold_letter) if gold_letter in LETTER_INDICES else 0
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

indo_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"indo_mmlu_it_{formation.lower()}",
        prompt_function=partial(get_indo_mmlu_it_prompt, language=Language.INDONESIAN if formation == "Native" else Language.ENGLISH),
        suite=("custom",),
        hf_repo="indolem/IndoMMLU",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for formation in ['Native', 'English']
]
TASKS_TABLE.extend(indo_mmlu_tasks)

# M3Exam
m3exam_tasks = [
    LightevalTaskConfig(
        name=f"m3exam_{language}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line['question'],
                "choices": line['choices'],
                "gold_idx": line['choices'].index(line['answer'][0]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="SEACrowd/m3exam",
        hf_subset=f"m3exam_{language}_seacrowd_qa",
        evaluation_splits=("test",),
        few_shots_split=None,
        hf_filter=partial(
            lambda line: len(line['answer']) > 0,
        ),
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
    for language in [
        'tha',
        'vie'
    ]
]
TASKS_TABLE.extend(m3exam_tasks)

# OpenAI-MMLU tasks
def get_openai_mmlu_it_prompt(line, task_name):
    instruction = BOXED_INSTRUCTION
    query = f"Question: {line['Question']}"
    option_texts = [line["A"], line["B"], line["C"], line["D"]]
    for i, choice in enumerate(option_texts):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold_letter = line["Answer"].upper() if isinstance(line["Answer"], str) else LETTER_INDICES[int(line["Answer"])]
    gold_index = LETTER_INDICES.index(gold_letter) if gold_letter in LETTER_INDICES else 0
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=option_texts,
        gold_index=gold_index,
    )

openai_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"openai_mmlu_it_{language[0].value}",
        prompt_function=get_openai_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="openai/MMMLU",
        hf_subset=language[1],
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        hf_revision="038c7808122969ead7456361af05cb8f47d247f8",
        generation_size=8192,
        stop_sequence=None,
        metric=[unified_boxed_metric],
    )
    for language in [
        (Language.ARABIC, "AR_XY"),
        (Language.BENGALI, "BN_BD"),
        (Language.GERMAN, "DE_DE"),
        (Language.SPANISH, "ES_LA"),
        (Language.FRENCH, "FR_FR"),
        (Language.INDONESIAN, "ID_ID"),
        (Language.ITALIAN, "IT_IT"),
        (Language.JAPANESE, "JA_JP"),
        (Language.KOREAN, "KO_KR"),
        (Language.PORTUGUESE, "PT_BR"),
        (Language.CHINESE, "ZH_CN"),
    ]
]
TASKS_TABLE.extend(openai_mmlu_tasks)

# TyDiQA tasks
tydiqa_tasks = [
    LightevalTaskConfig(
        name=f"tydiqa_{language.value}",
        prompt_function=get_qa_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("custom",),
        hf_repo="google-research-datasets/tydiqa",
        hf_subset="secondary_task",
        evaluation_splits=("validation",),
        hf_filter=partial(
            lambda language, line: line["id"].split("-")[0] == str(language).split(".")[1].lower(),
            language,
        ),
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(language, "prefix"),
            multilingual_quasi_f1_score_metric(language),
        ),
    )
    for language in [
        Language.ENGLISH,
        Language.ARABIC,
        Language.BENGALI,
        Language.INDONESIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.RUSSIAN,
        Language.THAI,
    ]
]
TASKS_TABLE.extend(tydiqa_tasks)

langcode2lang = {
    "ita_Latn": "Italian",
    "pol_Latn": "Polish",
    "ces_Latn": "Czech",
    "fra_Latn": "French",
    "por_Latn": "Portuguese",
    "jpn_Jpan": "Japanese",
    "ukr_Cyrl": "Ukrainian",
    "zho_Hans": "Chinese",
    "hun_Latn": "Hungarian",
    "kor_Hang": "Korean",
    "ron_Latn": "Romanian",
    "urd_Arab": "Urdu",
    "ben_Beng": "Bengali",
    "deu_Latn": "German",
    "rus_Cyrl": "Russian",
    "zsm_Latn": "Malay",
    "arb_Arab": "Arabic",
    "ell_Grek": "Greek",
    "spa_Latn": "Spanish",
    "tha_Thai": "Thai",
    "vie_Latn": "Vietnamese",
    "ind_Latn": "Indonesian",
    "nld_Latn": "Dutch",
    "heb_Hebr": "Hebrew",
    "kaz_Cyrl": "Kazakh",
    "tur_Latn": "Turkish",
    "eng_Latn": "English",
}

def get_translation_prompt_function(line, task_name):
    source_language, target_language = task_name.split(":")[1].split("-")
    instruction = f"Translate the following sentence into {langcode2lang[target_language]}"
    source_text = line[f"sentence_{source_language}"]
    target_text = line[f"sentence_{target_language}"]
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=f"{instruction}\n{langcode2lang[source_language]}: {source_text}\n{langcode2lang[target_language]}: ",
        choices=[target_text],
        gold_index=0,
    )

flores200_tasks = [
    LightevalTaskConfig(
        name=f"flores200:eng_Latn-{language}" if en_xx else f"flores200:{language}-eng_Latn",
        prompt_function=get_translation_prompt_function,
        suite=("custom",),
        hf_repo="facebook/flores",
        hf_subset=f"eng_Latn-{language}" if en_xx else f"{language}-eng_Latn",
        hf_avail_splits=["dev", "devtest"],
        evaluation_splits=["devtest"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=8192,
        # Metrics.bleu, Metrics.bleu_1, Metrics.bleu_4
        metric=[Metrics.chrf_plus],
        stop_sequence=["\n"],
        trust_dataset=True,
        version=0,
    )
    for en_xx in [True, False]
    for language in [
        "ita_Latn",
        "pol_Latn",
        "ces_Latn",
        "fra_Latn",
        "por_Latn",
        "jpn_Jpan",
        "ukr_Cyrl",
        "zho_Hans",
        "hun_Latn",
        "kor_Hang",
        "ron_Latn",
        "urd_Arab",
        "ben_Beng",
        "deu_Latn",
        "rus_Cyrl",
        "zsm_Latn",
        "arb_Arab",
        "ell_Grek",
        "spa_Latn",
        "tha_Thai",
        "vie_Latn",
        "ind_Latn",
        "nld_Latn",
        "heb_Hebr",
        "kaz_Cyrl",
        "tur_Latn",
    ]
]
TASKS_TABLE.extend(flores200_tasks)

# TriviaQA tasks
def get_triviaqa_boxed_prompt(line, task_name=None):
    q = (line.get("question") or line.get("Question") or "").strip()
    a = line.get("answer") or line.get("answers")
    if isinstance(a, list):
        a = a[0] if a else ""
    a = (a or "").strip()
    return Doc(
        task_name=task_name,
        query=f"Question: {q}\n{BOXED_INSTRUCTION}",
        choices=[a],
        gold_index=0,
    )

triviqa_tasks = [
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function=get_triviaqa_boxed_prompt,
        suite=("custom",),
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        generation_size=8192,
        stop_sequence=("\n",),
        metric=[unified_boxed_metric],
        few_shots_split=None,
    )
]
TASKS_TABLE.extend(triviqa_tasks)


# BBH tasks
def bbh_cot_prompt(line, task_name):
    subtask = task_name.split(":")[1]
    with open("prompts/bbh.json") as f:
        bbh_nshot_prompt = json.load(f)[subtask]
    des = bbh_nshot_prompt['description']
    query_template = bbh_nshot_prompt['query_template']
    query = des + "\n\n" + query_template.format(input=line["input"]) + "\n" + BOXED_INSTRUCTION
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["target"]],
        gold_index=0,
    )


# fmt: off
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
# fmt: on

bbh_tasks = [
    LightevalTaskConfig(
        name=f"bbh_cot:{subset}",
        prompt_function=bbh_cot_prompt,
        suite=["custom"],
        hf_repo="lighteval/big_bench_hard",
        hf_subset=subset,
        hf_revision="80610173426f05e6f1448f047e2db4840a7dd899",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        metric=[unified_boxed_metric],
        stop_sequence=["Q:"],
    )
    for subset in BBH_SUBSETS
]
TASKS_TABLE.extend(bbh_tasks)

# MATH tasks
def get_math_boxed_prompt(line, task_name=None):
    problem = (line.get("problem") or line.get("Problem") or "").strip()
    solution = line.get("solution") or line.get("Solution") or ""
    gold = extract_boxed(solution) if solution else ""
    if not gold:
        gold = solution.strip() if isinstance(solution, str) else ""
    return Doc(
        task_name=task_name,
        query=problem + "\n" + BOXED_INSTRUCTION,
        choices=[gold],
        gold_index=0,
    )

math_tasks = [
    LightevalTaskConfig(
        name=f"math:{config}",
        suite=("custom",),
        prompt_function=get_math_boxed_prompt,
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
        hf_subset=config,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        generation_size=8192,
        metric=[unified_boxed_math_metric],
        stop_sequence=None,
        trust_dataset=True,
        version=0,
    )
    for config in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]
TASKS_TABLE.extend(math_tasks)

polymath_prompt = {
    "en": "Note: Please put the final answer in the $\\boxed\{\}$.",
    "zh": "注意：请将最终答案放在 $\\boxed\{\}$ 中。",
    "ar": "ملاحظة: يُرجى وضع الإجابة النهائية في $\\boxed\{\}$.",
    "bn": "বিঃদ্রঃ: অনুগ্রহ করে চূড়ান্ত উত্তরটি $\\boxed\{\}$ এর মধ্যে রাখুন।",
    "de": "Hinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed\{\}$.",
    "es": "Nota: Por favor, coloque la respuesta final en el $\\boxed\{\}$.",
    "fr": "Remarque : Veuillez mettre la réponse finale dans le $\\boxed\{\}$.",
    "id": "Catatan: Silakan letakkan jawaban akhir di dalam $\\boxed\{\}$.",
    "it": "Nota: Per favore, metti la risposta finale nel $\\boxed\{\}$.",
    "ja": "注意：最終的な答えを $\\boxed\{\}$ に入れてください。",
    "ko": "참고: 최종 답안을 $\\boxed\{\}$ 안에 넣어 주세요.",
    "ms": "Nota: Sila letakkan jawapan akhir dalam $\\boxed\{\}$.",
    "pt": "Nota: Por favor, coloque a resposta final no $\\boxed\{\}$.",
    "ru": "Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed\{\}$.",
    "sw": "Kumbuka: Tafadhali weka jibu la mwisho katika $\\boxed\{\}$.",
    "te": "గమనిక: దయచేసి తుది జవాబును $\\boxed\{\}$ లో ఉంచండి.",
    "th": "หมายเหตุ: กรุณาใส่คำตอบสุดท้ายใน $\\boxed\{\}$.",
    "vi": "Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\\boxed\{\}$.",
}

def get_gsm8k_prompt(line, task_name: str = None):
    DELIM = "####"
    prompt = "Please reason step by step, and put your final answer within \\boxed\{\}."
    return Doc(
        task_name=task_name, 
        query=f"{line['question']}\n{prompt}",
        choices=[line["answer"].split(DELIM).pop().strip()], 
        gold_index=0
    )

# GSM8K tasks
gsm8k_tasks = [
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function=get_gsm8k_prompt,
        suite=("custom",),
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[unified_boxed_math_metric],
        generation_size=8192,
        stop_sequence=None,
        few_shots_split=None,
    )
]
TASKS_TABLE.extend(gsm8k_tasks)

def get_mgsm_prompt(line, task_name: str = None, lang: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['question']}\n{BOXED_INSTRUCTION}",
        choices=[str(line['answer_number'])],
        gold_index=0
    )

lang2prompt = {
    Language.ENGLISH.value: partial(get_mgsm_prompt, lang="en"),
    Language.FRENCH.value: partial(get_mgsm_prompt, lang="fr"),
    Language.GERMAN.value: partial(get_mgsm_prompt, lang="de"),
    Language.RUSSIAN.value: partial(get_mgsm_prompt, lang="ru"),
    Language.CHINESE.value: partial(get_mgsm_prompt, lang="zh"),
    Language.JAPANESE.value: partial(get_mgsm_prompt, lang="ja"),
    Language.THAI.value: partial(get_mgsm_prompt, lang="th"),
    Language.BENGALI.value: partial(get_mgsm_prompt, lang="bn"),
    Language.SPANISH.value: partial(get_mgsm_prompt, lang="es"),
}

# MGSM: juletxara/mgsm 使用 dataset script (mgsm.py)。若报错 "Dataset scripts are no longer supported"，
# 请安装 datasets 3.x：pip install 'datasets>=3.0,<4.0'
# datasets 3.x 下需 trust_dataset=True，否则会报 "set trust_remote_code=True"
mgsm_tasks = [
    LightevalTaskConfig(
        name=f"mgsm:{language.value}",
        prompt_function=lang2prompt[language.value],
        suite=("custom",),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        hf_avail_splits=["train", "test"],
        evaluation_splits=("test",),
        few_shots_split=None,
        metric=[unified_boxed_math_metric],
        generation_size=8192,
        stop_sequence=None,
        trust_dataset=True,
    )
    for language in [
        Language.ENGLISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.RUSSIAN,
        Language.CHINESE,
        Language.JAPANESE,
        Language.THAI,
        Language.BENGALI,
        Language.SPANISH,
    ]
]
TASKS_TABLE.extend(mgsm_tasks)

def get_polymath_prompt(line, task_name: str = None):
    lang = line['id'].split("-")[1]
    return Doc(
        task_name=task_name, 
        query=f"{line['question']}\n{BOXED_INSTRUCTION}",
        choices=[line['answer']], 
        gold_index=0
    )

# polymath tasks
polymath_tasks = [
    LightevalTaskConfig(
        name=f"polymath_{subset}:{language.value}",
        prompt_function=get_polymath_prompt,
        suite=("custom",),
        hf_repo="Qwen/PolyMath",
        hf_subset=standardize_tag(language.value),
        hf_avail_splits=['top', 'high', 'medium', 'low'],
        evaluation_splits=(subset,),
        few_shots_split=None,
        metric=[unified_boxed_math_metric],
        generation_size=8192,
        stop_sequence=None,
    )
    for subset in ['top', 'high', 'medium', 'low']
    for language in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CHINESE,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.MALAY,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.THAI,
        Language.VIETNAMESE,
    ]
]
TASKS_TABLE.extend(polymath_tasks)

# #---------------------
# # INSTRUCT MODEL EVALS
# #---------------------
# REASONING_TAG_PAIRS = [
#     ("<think>", "</think>"),
# ]

# ###########
# # MIXEVAL #
# ###########
# """The main differences between this implementation and LightEval's is that:

# - we don't use GPT-3.5-Turbo among the list of judges, but instead use flowaicom/Flow-Judge-v0.1.
# - we strip out the reasoning block from the predictions before passing them to the judge.
# - we left-truncate the predictions to fit within the 2048 token limit of the judge model.
# - we specify max_tokens=6144 for the judge to fit within the max context size of 8192 tokens.
# """

# MIXEVAL_EASY_TASKS_LIST = ",".join(["mixeval_easy:freeform", "mixeval_easy:multichoice"])
# MIXEVAL_HARD_TASKS_LIST = ",".join(["mixeval_hard:freeform", "mixeval_hard:multichoice"])

# MAX_INPUT_TOKENS = 2048  # LightEval judges have max_model_len=8192 and require space for a long judge prompt. We allow 2048 tokens for the prediction to fit within the limit.


# class JudgeLLMMixEval(JudgeLLM):
#     def compute(self, sample_ids: list[str], responses: list, formatted_docs: list[Doc], **kwargs) -> dict[str, float]:
#         """
#         Compute the score of a generative task using a llm as a judge.
#         The generative task can be multiturn with 2 turns max, in that case, we
#         return scores for turn 1 and 2. Also returns user_prompt and judgement
#         which are ignored later by the aggregator.
#         """
#         questions = [formatted_doc.specific["question"] for formatted_doc in formatted_docs]
#         options = [formatted_doc.choices for formatted_doc in formatted_docs]
#         golds = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]

#         predictions = []
#         num_truncated = 0
#         for response in responses:
#             prediction_text = remove_reasoning_tags(response[0].result[0], tag_pairs=REASONING_TAG_PAIRS).strip()

#             # Left-truncate the prediction to fit within the max input tokens for the judge model.
#             if len(response[0].generated_tokens[0]) > MAX_INPUT_TOKENS:
#                 # One token is worth ~4 characters, so we estimate the number of characters to truncate.
#                 prediction_text = f"{prediction_text[-MAX_INPUT_TOKENS * 4 :]}"

#             predictions.append(prediction_text)

#         print(f"Number of predictions truncated to fit within {MAX_INPUT_TOKENS} tokens: {num_truncated}")  # noqa: T201

#         scores, messages, judgements = self.judge.evaluate_answer_batch(questions, predictions, options, golds)

#         metrics = []
#         for i in range(len(sample_ids)):
#             metrics.append(
#                 {
#                     f"judge_score_{self.short_judge_name}": scores[i],
#                     f"user_prompt_{self.short_judge_name}": messages[i],
#                     f"judgement_{self.short_judge_name}": judgements[i],
#                 }
#             )

#         return metrics


# llm_judge_mixeval_multichoice_flow_judge = SampleLevelMetricGrouping(
#     metric_name=["llm_judge_mixeval_flow"],
#     higher_is_better={"judge_score_flow": True},
#     category=MetricCategory.LLM_AS_JUDGE,
#     use_case=MetricUseCase.SUMMARIZATION,
#     sample_level_fn=JudgeLLMMixEval(
#         judge_model_name="flowaicom/Flow-Judge-v0.1",
#         template=flow_judge_for_multichoice_template,
#         process_judge_response=process_judge_response,
#         judge_backend="vllm",
#         short_judge_name="flow",
#         max_tokens=1024,  # Flow judge has a context length limit of 8192 tokens and 2048 are reserved for the input
#     ).compute,
#     corpus_level_fn={
#         "judge_score_flow": np.mean,
#     },
# )


# llm_judge_mixeval_freeform_flow_judge = SampleLevelMetricGrouping(
#     metric_name=["llm_judge_mixeval_flow"],
#     higher_is_better={"judge_score": True},
#     category=MetricCategory.LLM_AS_JUDGE,
#     use_case=MetricUseCase.SUMMARIZATION,
#     sample_level_fn=JudgeLLMMixEval(
#         judge_model_name="flowaicom/Flow-Judge-v0.1",
#         template=flow_judge_for_freeform_template,
#         process_judge_response=process_judge_response,
#         judge_backend="vllm",
#         short_judge_name="flow",
#         max_tokens=1024,  # Flow judge has a context length limit of 8192 tokens and 2048 are reserved for the input
#     ).compute,
#     corpus_level_fn={
#         "judge_score_flow": mean_dv_5,
#     },
# )

# mixeval_freeform_easy = LightevalTaskConfig(
#     name="mixeval_easy:freeform",
#     prompt_function=mixeval_freeform_prompt,
#     suite=["custom"],
#     hf_repo="MixEval/MixEval",
#     hf_subset="MixEval",
#     metric=[llm_judge_mixeval_freeform_flow_judge],
#     hf_avail_splits=["free_form"],
#     evaluation_splits=["free_form"],
#     few_shots_split=None,
#     few_shots_select="random_sampling",
#     generation_size=100,  # overridden at runtime by the generation parameters
#     stop_sequence=[],  # no stop sequence, will use eot token
#     version="0.1",
# )

# mixeval_multichoice_easy = LightevalTaskConfig(
#     name="mixeval_easy:multichoice",
#     prompt_function=mixeval_multichoice_prompt,
#     suite=["custom"],
#     hf_repo="MixEval/MixEval",
#     hf_subset="MixEval",
#     metric=[llm_judge_mixeval_multichoice_flow_judge],
#     hf_avail_splits=["multiple_choice"],
#     evaluation_splits=["multiple_choice"],
#     few_shots_split=None,
#     few_shots_select="random_sampling",
#     generation_size=100,  # overridden at runtime by the generation parameters
#     stop_sequence=[],  # no stop sequence, will use eot token
#     version="0.1",
# )

# mixeval_freeform_hard = LightevalTaskConfig(
#     name="mixeval_hard:freeform",
#     prompt_function=mixeval_freeform_prompt,
#     suite=["custom"],
#     hf_repo="MixEval/MixEval",
#     hf_subset="MixEval_Hard",
#     metric=[llm_judge_mixeval_multichoice_flow_judge],
#     hf_avail_splits=["free_form"],
#     evaluation_splits=["free_form"],
#     few_shots_split=None,
#     few_shots_select="random_sampling",
#     generation_size=100,  # overridden at runtime by the generation parameters
#     stop_sequence=[],  # no stop sequence, will use eot token
#     version="0.1",
# )


# mixeval_multichoice_hard = LightevalTaskConfig(
#     name="mixeval_hard:multichoice",
#     prompt_function=mixeval_multichoice_prompt,
#     suite=["custom"],
#     hf_repo="MixEval/MixEval",
#     hf_subset="MixEval_Hard",
#     metric=[llm_judge_mixeval_multichoice_flow_judge],
#     hf_avail_splits=["multiple_choice"],
#     evaluation_splits=["multiple_choice"],
#     few_shots_split=None,
#     few_shots_select="random_sampling",
#     generation_size=100,  # overridden at runtime by the generation parameters
#     stop_sequence=[],  # no stop sequence, will use eot token
#     version="0.1",
# )

# TASKS_TABLE.extend(
#     [
#         mixeval_multichoice_easy,
#         mixeval_freeform_easy,
#         mixeval_multichoice_hard,
#         mixeval_freeform_hard,
#     ]
# )


# ###########
# # GSMPlus #
# ###########
# def gsm_plus_prompt(line, task_name: str = None):
#     # Prompt template adapted from
#     # - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
#     # - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
#     # Note that it is important to have the final answer in a box for math-verify to work correctly
#     MATH_QUERY_TEMPLATE = """
# Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

# {Question}
# """.strip()

#     # Some prompts require critical thinking (around 1k/10k), we skip them as
#     # they are a bit trickier to eval with regular text extraction.
#     if line["perturbation_type"] == "critical thinking":
#         return None

#     return Doc(
#         task_name=task_name,
#         query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
#         choices=[line["answer"]],
#         gold_index=0,
#     )


# gsm_plus = LightevalTaskConfig(
#     name="gsm_plus",
#     suite=["custom"],
#     prompt_function=gsm_plus_prompt,
#     hf_repo="qintongli/GSM-Plus",
#     hf_subset="default",
#     hf_avail_splits=["testmini"],
#     evaluation_splits=["testmini"],
#     few_shots_split=None,
#     few_shots_select=None,
#     generation_size=None,
#     metric=[
#         Metrics.math_pass_at_1_1n,
#     ],
#     stop_sequence=None,
#     trust_dataset=True,
#     version=0,
# )

# TASKS_TABLE.append(gsm_plus)


# remove pmi_norm from all tasks to save on double inference
for task in TASKS_TABLE:
    task.metric = [
        metric
        for metric in task.metric
        if metric.category != MetricCategory.MULTICHOICE_PMI
    ]

if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
