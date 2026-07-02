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

lighteval vllm "model_name=HuggingFaceTB/SmolLM3-3B,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" \
    "custom|gsm_plus|0|0,custom|mixeval_hard|0|0" \
    --use-chat-template \
    --output-dir evals/ \
    --custom-tasks tasks.py \
    --save-details
"""
from dis import Instruction
from functools import partial
import numpy as np
import json
from typing import Callable
import logging

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
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    SampleLevelMetric,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import winogrand_adapter
from lighteval.tasks.multilingual.tasks import TASKS_TABLE as ML_TASKS_TABLE
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language

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
    num_samples=8,
) -> SampleLevelMetric:

    def sample_level_fn(golds: list[str], predictions: list[str], formatted_doc: Doc) -> float:
        def extract_target_from_pred(pred):
            if '\n' in pred:
                pred = pred.split('\n')[0]
            try:
                pred = eval(pred)
                return [pred["answer"]]
            except:
                for letter in LETTER_INDICES:
                    if pred.startswith(f"{letter}.") or f"{letter}." in pred or letter == pred:
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
        metric_name=f"extractive_match_avg@{num_samples}",
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

def multilingual_extractive_match_metric_avg_k(
    language,
    gold_extraction_target,
    pred_extraction_target,
    aggregation_function,
    fallback_mode,
    precision,
    num_samples=8,
):
    latex_gold_metric = multilingual_extractive_match_metric(
        language=language,
        fallback_mode=fallback_mode,
        precision=precision,
        gold_extraction_target=gold_extraction_target,
        pred_extraction_target=pred_extraction_target,
        aggregation_function=aggregation_function,
    )
    latex_gold_metric.metric_name = f"extractive_match_avg@{num_samples}"
    latex_gold_metric.category = MetricCategory.GENERATIVE_SAMPLING
    return latex_gold_metric

multiple_choice_metric = multiple_choice_extractive_match_metric(num_samples=1)
multiple_choice_metric_avg_8 = multiple_choice_extractive_match_metric(num_samples=8)
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
latex_gold_metric_avg_8 = multilingual_extractive_match_metric_avg_k(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(boxed_match_priority=0),
    ),
    aggregation_function=np.mean,
    num_samples=8,
)

mcq_step_by_step_prompts = {
    Language.ARABIC: "يرجى شرح الحل خطوة بخطوة، ووضع إجابتك النهائية داخل المربع \\boxed{}",
    Language.BENGALI: "অনুগ্রহ করে ধাপে ধাপে যুক্তি দিন এবং আপনার উত্তর \\boxed{} এর মধ্যে লিখুন।",
    Language.CHINESE: "请逐步展开推理，并将最终答案放在\\boxed{}内。",
    Language.DUTCH: "Leg je redenering stap voor stap uit en plaats je eindantwoord tussen \\boxed{}.",
    Language.FRENCH: "Veuillez raisonner étape par étape et inscrire votre réponse finale dans \\boxed{}.",
    Language.GERMAN: "Bitte begründen Sie Ihre Antwort Schritt für Schritt und geben Sie Ihr Endergebnis innerhalb von \\boxed{} an.",
    Language.GREEK: "Παρακαλώ συλλογιστείτε βήμα προς βήμα και βάλτε την τελική σας απάντηση εντός \\boxed{}.",
    Language.HEBREW: "אנא נמק שלב אחר שלב, וסמן את תשובתך הסופית בתוך \\boxed{}.",
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
    Language.ENGLISH: "Please reason step by step, and put your final answer within \\boxed{}.",
    Language.CZECH: "Prosím, zdůvodněte to krok za krokem a konečnou odpověď uveďte do rámečku \\boxed{}.",
    Language.ROMANIAN: "Te rog să argumentezi pas cu pas și să scrii răspunsul final în \\boxed{}.",
    Language.THAI: "กรุณาอธิบายขั้นตอนการคิดอย่างเป็นเหตุเป็นผล และแสดงคำตอบสุดท้ายภายใน \\boxed{}.",
    Language.AZERBAIJANI: "Zəhmət olmasa, addım-addım əsaslandırın və son cavabınızı \\boxed{} bölməsinə qoyun",
    Language.NEPALI: "कृपया चरण दर चरण तर्क गर्नुहोस्, र अन्तिम उत्तर भेट्टाएर राख्नुहोस् \\boxed{}",
}
iso_convert = {
    "ita_Latn": Language.ITALIAN,
    "pol_Latn": Language.POLISH,
    "ces_Latn": Language.CZECH,
    "fra_Latn": Language.FRENCH,
    "por_Latn": Language.PORTUGUESE,
    "jpn_Jpan": Language.JAPANESE,
    "ukr_Cyrl": Language.UKRAINIAN,
    "zho_Hans": Language.CHINESE,
    "hun_Latn": Language.HUNGARIAN,
    "kor_Hang": Language.KOREAN,
    "ron_Latn": Language.ROMANIAN,
    "urd_Arab": Language.URDU,
    "ben_Beng": Language.BENGALI,
    "deu_Latn": Language.GERMAN,
    "rus_Cyrl": Language.RUSSIAN,
    "zsm_Latn": Language.MALAY,
    "arb_Arab": Language.ARABIC,
    "ell_Grek": Language.GREEK,
    "spa_Latn": Language.SPANISH,
    "eng_Latn": Language.ENGLISH,
    "tha_Thai": Language.THAI,
    "vie_Latn": Language.VIETNAMESE,
    "ind_Latn": Language.INDONESIAN,
    "nld_Latn": Language.DUTCH,
    "heb_Hebr": Language.HEBREW,
    "kaz_Cyrl": Language.KAZAKH,
    "tur_Latn": Language.TURKISH,
    "azj_Latn": Language.AZERBAIJANI,
    "npi_Deva": Language.NEPALI,
}

# custom|boolq_it|0|0
# custom|commonsenseqa_it|0|0
# custom|openbookqa_it|0|0
# custom|piqa_it|0|0
# custom|siqa_it|0|0
def get_arc_it_prompt(line, task_name):
    # instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    instruction = "Please show your choice within \\boxed{} with only the choice letter, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    for i, choice in enumerate(line["choices"]["text"]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = LETTER_INDICES[int(line["answerKey"]) - 1] if line["answerKey"].isdigit() else line["answerKey"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        few_shots_split="train",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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
        few_shots_split="train",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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
    instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    query = f"{line['passage']}\nQuestion: {line['question']}"
    for i, choice in enumerate(['Yes', 'No']):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = 'A' if line['answer'] else 'B'
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

# BoolQ task
boolq_task = LightevalTaskConfig(
    name="boolq_it",
    prompt_function=get_boolq_it_prompt,
    suite=("custom",),
    hf_repo="google/boolq",
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    generation_size=10,
    stop_sequence=None,
    metric=[multiple_choice_metric],
)
TASKS_TABLE.append(boolq_task)

# HellaSwag tasks
def get_hellaswag_it_prompt(line, task_name):
    prompt_function = get_hellaswag_prompt_function(
        language=Language.ENGLISH,
        adapter=lambda line: {
            "activity_label": line["activity_label"],
            "ctx_a": line["ctx_a"],
            "ctx_b": line["ctx_b"],
            "continuations": line["endings"],
            "gold_idx": int(line["label"]),
        },
        formulation=MCFFormulation(),
    )
    prompt = prompt_function(line, task_name)
    if prompt is None:
        return None
    query = """You are given a context and four possible endings. Choose the ending that best completes the context.

Context: {text}""".format(text=prompt.query.split("\nAnswer:")[0].strip())
    # instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    return Doc(
        task_name=prompt.task_name,
        query=query + "\n" + instruction,
        choices=[prompt.choices[prompt.gold_index[0]].strip()],
        gold_index=0,
    )

hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_it",
        suite=["custom"],
        prompt_function=get_hellaswag_it_prompt,
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        evaluation_splits=["validation"],
        hf_avail_splits=["train", "validation"],
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
    )
]
TASKS_TABLE.extend(hellaswag_tasks)

mlmm_hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag_it_{lang.value}",
        suite=["custom"],
        prompt_function=get_hellaswag_it_prompt,
        hf_repo="jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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
TASKS_TABLE.extend(mlmm_hellaswag_tasks)

def get_commonsense_qa_it_prompt(line, task_name):
    instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    query = f"Question: {line['question']}"
    for label, choice in zip(line['choices']['label'], line['choices']['text']):
        query += f"\n{label}. {choice}"
    gold = line['answerKey']
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        few_shots_split="train",
        generation_size=10,
        stop_sequence=None,
        metric=[multiple_choice_metric],
    )
]
TASKS_TABLE.extend(commonsense_qa_tasks)

def get_openbook_qa_it_prompt(line, task_name):
    instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    query = f"Question: {line['question_stem']}"
    for label, choice in zip(line['choices']['label'], line['choices']['text']):
        query += f"\n{label}. {choice}"
    gold = line['answerKey']
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        few_shots_split="train",
        generation_size=10,
        stop_sequence=None,
        metric=[multiple_choice_metric],
    )
]
TASKS_TABLE.extend(openbook_qa_tasks)

# Winogrande tasks
def get_winogrande_it_prompt(line, task_name):
    prompt_function = get_continuation_prompt_function(
        Language.ENGLISH,
        partial(winogrand_adapter, Language.ENGLISH),
        formulation=MCFFormulation(),
    )
    prompt = prompt_function(line, task_name)
    query = """Choose the option that best completes the sentence.

Sentence: {text}""".format(text=prompt.query.split("\nAnswer:")[0].strip())
    instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    # instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[prompt.choices[prompt.gold_index[0]].strip()],
        gold_index=0,
    )

winogrande_tasks = [
    LightevalTaskConfig(
        name=f"winogrande_it",
        suite=("custom",),
        prompt_function=get_winogrande_it_prompt,
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        trust_dataset=True,
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        generation_size=10,
        stop_sequence=None,
        metric=[multiple_choice_metric],
    )
]
TASKS_TABLE.extend(winogrande_tasks)

def get_piqa_it_prompt(line, task_name):
    instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    query = f"Question: {line['goal']}"
    for i, choice in enumerate([line["sol1"], line["sol2"]]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = LETTER_INDICES[int(line["label"])]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        few_shots_split="train",
        generation_size=10,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
    )
]
TASKS_TABLE.extend(piqa_tasks)

# Global PIQA tasks
def get_global_piqa_it_prompt(line, task_name):
    # # Qwen3-1.7B: avg@8 - 0.7426
    # instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    # Qwen3-1.7B: avg@8 - 0.5730
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    # # Qwen3-1.7B: 64.1
    # instruction = "Please directly show your choice letter within \\boxed{} without any other content."
    # Qwen3-1.7B: 64.2
    # instruction = "Please show your choice in \\boxed{} with only the choice letter, e.g., \\boxed{A}."
    query = f"Question: {line['prompt']}"
    for i, choice in enumerate([line["solution0"], line["solution1"]]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = LETTER_INDICES[int(line["label"])]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

global_piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_it_{language}",
        prompt_function=get_global_piqa_it_prompt,
        suite=["custom"],
        hf_repo="mrlbenchmarks/global-piqa-nonparallel",
        hf_subset=language,
        trust_dataset=True,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
    )
    for language in [
        "ita_latn",
        "pol_latn",
        "ces_latn",
        "fra_latn_fran",
        "fra_latn_cana",
        "por_latn_braz",
        "por_latn_port",
        "jpn_jpan",
        "ukr_cyrl",
        "cmn_hans",
        "cmn_hant",
        "hun_latn",
        "kor_hang",
        "ron_latn",
        "urd_arab",
        "ben_beng",
        "deu_latn",
        "rus_cyrl",
        "zsm_latn",
        "arb_arab",
        "ell_grek",
        "spa_latn_mexi",
        "spa_latn_peru",
        "spa_latn_spai",
        "tha_thai",
        "vie_latn",
        "ind_latn",
        "nld_latn",
        "heb_hebr",
        "kaz_cyrl",
        "tur_latn",
        "azj_latn",
        "npi_deva",
        "eng_latn",
        "amh_ethi",
        "cat_latn",
        "ekk_latn",
        "tgl_latn",
        "guj_gujr",
        "hrv_latn",
        "jav_latn",
        "nob_latn",
        "slk_latn",
        "srp_cyrl",
        "tam_taml",
        "yor_latn",
        "bul_cyrl",
        "fin_latn",
        "glg_latn",
        "hin_deva",
        "ibo_latn",
        "lit_latn",
        "mar_deva",
        "pan_guru",
        "slv_latn",
        "swh_latn",
        "tel_telu",
        "zul_latn",
        "swe_latn",
        "pes_arab",
    ]
]
TASKS_TABLE.extend(global_piqa_tasks)

def get_siqa_it_prompt(line, task_name):
    instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    query = f"{line['context']}\nQuestion: {line['question']}"
    for i, choice in enumerate([line["answerA"], line["answerB"], line["answerC"]]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = LETTER_INDICES[int(line["label"]) - 1]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        few_shots_split="train",
        generation_size=10,
        stop_sequence=None,
        metric=[multiple_choice_metric],
    )
]
TASKS_TABLE.extend(siqa_tasks)

# MMLU tasks
# fmt: off
MMLU_SUBSETS = [
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
def get_mmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    for i, choice in enumerate(line["choices"]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = LETTER_INDICES[int(line["answer"])]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        few_shots_split="dev",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
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
        few_shots_split="test",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
    )
    for subset in MMLU_SUBSETS
]
TASKS_TABLE.extend(mmlu_redux_tasks)

def get_agi_eval_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    options = [option[3:] for option in line["options"]]
    for i, choice in enumerate(options):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["label"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

agi_eval_tasks = [
    LightevalTaskConfig(
        name=f"agi_eval_it:{subset}",
        prompt_function=get_agi_eval_it_prompt,
        suite=("custom",),
        hf_repo="lighteval/agi_eval_en",
        hf_subset=subset,
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split="validation",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
    )
    for subset in [
        'aqua_rat', 
        'logiqa-en', 
        'lsat-ar', 
        'lsat-lr', 
        'lsat-rc', 
        'sat-en', 
        'sat-math'
    ]
]
TASKS_TABLE.extend(agi_eval_tasks)

def get_mmlu_pro_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
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
        few_shots_split="validation",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
        version=0,
    )
]
TASKS_TABLE.extend(mmlu_pro_it_tasks)

# MMLU-ProX-Lite IT tasks
mmlu_pro_x_lite_it_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_x_lite_it_{language.value}",
        prompt_function=get_mmlu_pro_it_prompt,
        suite=("custom",),
        hf_repo="li-lab/MMLU-ProX-Lite",
        hf_subset=standardize_tag(language.value),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
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

def get_gpqa_it_prompt_function(line, task_name):
    import re
    import random

    def preprocess(text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    
    choices = [
        preprocess(line["Incorrect Answer 1"]),
        preprocess(line["Incorrect Answer 2"]),
        preprocess(line["Incorrect Answer 3"]),
    ]
    random.shuffle(choices)
    correct_answer = preprocess(line["Correct Answer"])
    correct_answer_index = random.randint(0, len(choices))
    choices.insert(correct_answer_index, correct_answer)

    query = f"Question: {line['Question']}"
    for i in range(len(choices)):
        query += f"\n{LETTER_INDICES[i]}. {choices[i]}"
    gold = LETTER_INDICES[correct_answer_index]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

# GPQA tasks
gpqa_tasks = [
    LightevalTaskConfig(
        name=f"gpqa_it",
        prompt_function=get_gpqa_it_prompt_function,
        suite=("custom",),
        hf_repo="nmayorga7/gpqa_diamond",
        hf_subset="default",
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split="train",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
        version=0,
    )
]
TASKS_TABLE.extend(gpqa_tasks)

# multilingual GPQA
mgpqa_tasks = [
    LightevalTaskConfig(
        name=f"gpqa_it_{language}",
        prompt_function=get_gpqa_it_prompt_function,
        suite=("custom",),
        hf_repo="LLaMAX/BenchMAX_Science",
        hf_subset=language,
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split="train",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
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
TASKS_TABLE.extend(mgpqa_tasks)

def get_super_gpqa_it_prompt_function(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    
    query = f"Question: {line['question']}"
    choices = line['options']
    for i in range(len(choices)):
        query += f"\n{LETTER_INDICES[i]}. {choices[i]}"
    query += "Answer: Let's think step by step:"
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[line["answer_letter"]],
        gold_index=0,
    )

# super gpqa tasks
supergpqa_tasks = [
    LightevalTaskConfig(
        name=f"supergpqa_it",
        prompt_function=get_super_gpqa_it_prompt_function,
        suite=("custom",),
        hf_repo="m-a-p/SuperGPQA",
        hf_subset="default",
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split="train",
        generation_size=8192,
        metric=[latex_gold_metric_avg_8],
        stop_sequence=None,
        version=0,
    )
]
TASKS_TABLE.extend(supergpqa_tasks)

def get_belebele_it_prompt(line, task_name):
    # instruction = "Please show your choice in the `answer` field with only the choice letter, e.g., {\"answer\": \"C\"}."
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    
    query = f"{line['flores_passage']}\nQuestion: {line['question']}"
    for i in range(1, 5):
        query += f"\n{LETTER_INDICES[i - 1]}. {line[f'mc_answer{i}']}"
    gold = LETTER_INDICES[int(line["correct_answer_num"]) - 1]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        metric=[latex_gold_metric_avg_8],
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
        "azj_Latn",
        "npi_Deva",
    ]
]
TASKS_TABLE.extend(belebele_tasks)


def get_include_it_prompt(line, task_name):
    # instruction = mcq_step_by_step_prompts[language]
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    for i, choice in enumerate([line["option_a"], line["option_b"], line["option_c"], line["option_d"]]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = LETTER_INDICES[line["answer"]]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

# INCLUDE tasks
include_it_tasks = [
    LightevalTaskConfig(
        name=f"include_it_{language.value}",
        prompt_function=get_include_it_prompt,
        suite=("custom",),
        hf_repo="CohereLabs/include-base-44",
        hf_subset=str(language).split(".")[1].capitalize(),
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        few_shots_split="validation",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric_avg_8],
    )
    for language in [
        Language.ARABIC,
        Language.AZERBAIJANI,
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
        Language.NEPALI,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SPANISH,
        Language.TURKISH,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
    ]
]
TASKS_TABLE.extend(include_it_tasks)

def get_global_mmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    for i, choice in enumerate([line["option_a"], line["option_b"], line["option_c"], line["option_d"]]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["answer"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        few_shots_split="dev",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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
        Language.TURKISH,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
    ]
]
TASKS_TABLE.extend(global_mmlu_tasks)

# -------------------- Language Specific MMLU tasks --------------------
def get_cmmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['Question'] if 'Question' in line else line['question']}"
    for i, choice in enumerate([line["A"], line["B"], line["C"], line["D"]]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["Answer"] if "Answer" in line else line["answer"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

cmmlu_tasks = [
    LightevalTaskConfig(
        name=f"zho_mmlu_it:{subset}",
        prompt_function=get_cmmlu_it_prompt,
        suite=("custom",),
        hf_repo="lmlmcat/cmmlu",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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

ceval_tasks = [
    LightevalTaskConfig(
        name=f"ceval_it:{subset}",
        prompt_function=get_cmmlu_it_prompt,
        suite=("custom",),
        hf_repo="ceval/ceval-exam",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
    )
    for subset in [
        "accountant",
        "advanced_mathematics",
        "art_studies",
        "basic_medicine",
        "business_administration",
        "chinese_language_and_literature",
        "civil_servant",
        "clinical_medicine",
        "college_chemistry",
        "college_economics",
        "college_physics",
        "college_programming",
        "computer_architecture",
        "computer_network",
        "discrete_mathematics",
        "education_science",
        "electrical_engineer",
        "environmental_impact_assessment_engineer",
        "fire_engineer",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_chinese",
        "high_school_geography",
        "high_school_history",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "ideological_and_moral_cultivation",
        "law",
        "legal_professional",
        "logic",
        "mao_zedong_thought",
        "marxism",
        "metrology_engineer",
        "middle_school_biology",
        "middle_school_chemistry",
        "middle_school_geography",
        "middle_school_history",
        "middle_school_mathematics",
        "middle_school_physics",
        "middle_school_politics",
        "modern_chinese_history",
        "operating_system",
        "physician",
        "plant_protection",
        "probability_and_statistics",
        "professional_tour_guide",
        "sports_science",
        "tax_accountant",
        "teacher_qualification",
        "urban_and_rural_planner",
        "veterinary_medicine"
    ]
]
TASKS_TABLE.extend(ceval_tasks)

# arabic mmlu tasks
def get_arabic_mmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"{line['Context']}\nQuestion: {line['Question']}"
    choices = [str(o) for o in [line[f"Option {i}"] for i in range(1, 6)] if o]
    for i, choice in enumerate(choices):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["Answer Key"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        metric=[latex_gold_metric],
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
def get_turkish_mmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    for i, choice in enumerate(line["choices"]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["answer"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

turkish_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"tur_mmlu_it:{subset}",
        prompt_function=get_turkish_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="AYueksel/TurkishMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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
]
TASKS_TABLE.extend(turkish_mmlu_tasks)

# KazMMLU tasks
def get_kaz_mmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['Question']}"
    choices = [line["Option A"], line["Option B"], line["Option C"], line["Option D"], line["Option E"]]
    for i, choice in enumerate(choices):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["Answer Key"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

kaz_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"kaz_mmlu_it:{subject}",
        prompt_function=get_kaz_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="MBZUAI/KazMMLU",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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

# GreekMMLU tasks
def get_greek_mmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    for i, choice in enumerate(line["choices"]):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = LETTER_INDICES[line["answer"]]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

greek_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"greek_mmlu_it:{subject}",
        prompt_function=get_greek_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="dascim/GreekMMLU",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
    )
    for subject in [
        "Accounting",
        "Agriculture_Professional",
        "Agriculture_University",
        "Art_Professional",
        "Art_Secondary_School",
        "Art_University",
        "Biology",
        "Chemistry",
        "Civil_Engineering",
        "Clinical_Knowledge",
        "Computer_Networks_and_Security",
        "Computer_Science_Professional",
        "Computer_Science_University",
        "Driving_Rules",
        "Economics_Professional",
        "Economics_University",
        "Education_Professional",
        "Education_University",
        "Electrical_Engineering",
        "General_Knowledge",
        "Geography_Primary_School",
        "Geography_Secondary_School",
        "Government_and_Politics_Primary_School",
        "Government_and_Politics_Secondary_School",
        "Greek_History_Primary_School",
        "Greek_History_Professional",
        "Greek_History_Secondary_School",
        "Greek_Literature",
        "Greek_Mythology",
        "Greek_Traditions",
        "Law",
        "Management_Professional",
        "Management_University",
        "Maritime_Safety_and_Rescue_Operations",
        "Mathematics",
        "Medicine_Professional",
        "Medicine_University",
        "Modern_Greek_Language_Primary_School",
        "Modern_Greek_Language_Secondary_School",
        "Physics_Primary_School",
        "Physics_Professional",
        "Physics_University",
        "Prehistory",
        "World_History",
        "World_Religions",
    ]
]
TASKS_TABLE.extend(greek_mmlu_tasks)

def get_indo_career_it_prompt(line, task_name, subject):
    instruction = f"This is a {subject} question for {line['Exam Type']}. " + "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['Question']}"
    choices = [line["Option A"], line["Option B"], line["Option C"], line["Option D"], line["Option E"]]
    for i, choice in enumerate(choices):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["Answer Key"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

# IndoCareer tasks
indo_career_tasks = [
    LightevalTaskConfig(
        name=f"indo_career_it:{subject}",
        prompt_function=partial(get_indo_career_it_prompt, subject=subject),
        suite=("custom",),
        hf_repo="indolem/IndoCareer",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
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
]
TASKS_TABLE.extend(indo_career_tasks)

def get_indo_culture_it_prompt(line, task_name):
    instruction = f"For the context {line['province']}, " + "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['context']}"
    choices = [option[3:] for option in eval(line["options"])]
    for i, choice in enumerate(choices):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["answer"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

# IndoCulture tasks
indo_culture_tasks = [
    LightevalTaskConfig(
        name=f"indo_culture_it",
        prompt_function=get_indo_culture_it_prompt,
        suite=("custom",),
        hf_repo="indolem/IndoCulture",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
    )
]
TASKS_TABLE.extend(indo_culture_tasks)

# IndoMMLU tasks
def get_indo_mmlu_it_prompt(line, task_name):
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['question']}"
    choices = [option[3:] for option in eval(line["options"])]
    for i, choice in enumerate(choices):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["answer"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
    )

indo_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"indo_mmlu_it",
        prompt_function=get_indo_mmlu_it_prompt,
        suite=("custom",),
        hf_repo="indolem/IndoMMLU",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        generation_size=8192,
        stop_sequence=None,
        metric=[latex_gold_metric],
    )
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
        few_shots_split=("validation"),
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
    instruction = "Please reason step by step, and put your final answer within \\boxed{}, e.g., \\boxed{C}."
    query = f"Question: {line['Question']}"
    choices = [line["A"], line["B"], line["C"], line["D"]]
    for i, choice in enumerate(choices):
        query += f"\n{LETTER_INDICES[i]}. {choice}"
    gold = line["Answer"]
    return Doc(
        task_name=task_name,
        query=query + "\n" + instruction,
        choices=[gold],
        gold_index=0,
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
        metric=[latex_gold_metric],
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
        few_shots_split="train",
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
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=8192,
        # Metrics.bleu, Metrics.bleu_1, Metrics.bleu_4
        metric=[Metrics.chrf_plus],
        stop_sequence=["\n"],
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
triviqa_tasks = [
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function=prompt.triviaqa,
        suite=("custom",),
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        generation_size=8192,
        stop_sequence=("\n",),
        metric=[Metrics.quasi_exact_match_triviaqa],
        few_shots_select="random_sampling_from_train",
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
    few_shots = [query_template.format(input=example['input']) + example['target'] for example in bbh_nshot_prompt['examples']]
    query = des + "\n\n" + "\n\n".join(few_shots) + "\n\n" + query_template.format(input=line["input"])

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
        few_shots_split="train",
        trust_dataset=True,
        generation_size=8192,
        metric=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Q:"],
    )
    for subset in BBH_SUBSETS
]
TASKS_TABLE.extend(bbh_tasks)

# MATH tasks
math_tasks = [
    LightevalTaskConfig(
        name=f"math:{config}",
        suite=("custom",),
        prompt_function=prompt.math_cot,
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
        hf_subset=config,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=8192,
        metric=[latex_gold_metric_avg_8],
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
    "en": "Note: Please put the final answer in the $\\boxed{}$.",
    "zh": "注意：请将最终答案放在 $\\boxed{}$ 中。",
    "ar": "ملاحظة: يُرجى وضع الإجابة النهائية في $\\boxed{}$.",
    "bn": "বিঃদ্রঃ: অনুগ্রহ করে চূড়ান্ত উত্তরটি $\\boxed{}$ এর মধ্যে রাখুন।",
    "de": "Hinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{}$.",
    "es": "Nota: Por favor, coloque la respuesta final en el $\\boxed{}$.",
    "fr": "Remarque : Veuillez mettre la réponse finale dans le $\\boxed{}$.",
    "id": "Catatan: Silakan letakkan jawaban akhir di dalam $\\boxed{}$.",
    "it": "Nota: Per favore, metti la risposta finale nel $\\boxed{}$.",
    "ja": "注意：最終的な答えを $\\boxed{}$ に入れてください。",
    "ko": "참고: 최종 답안을 $\\boxed{}$ 안에 넣어 주세요.",
    "ms": "Nota: Sila letakkan jawapan akhir dalam $\\boxed{}$.",
    "pt": "Nota: Por favor, coloque a resposta final no $\\boxed{}$.",
    "ru": "Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed{}$.",
    "sw": "Kumbuka: Tafadhali weka jibu la mwisho katika $\\boxed{}$.",
    "te": "గమనిక: దయచేసి తుది జవాబును $\\boxed{}$ లో ఉంచండి.",
    "th": "หมายเหตุ: กรุณาใส่คำตอบสุดท้ายใน $\\boxed{}$.",
    "vi": "Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\\boxed{}$.",
}

def get_gsm8k_prompt(line, task_name: str = None):
    DELIM = "####"
    prompt = "Please reason step by step, and put your final answer within \\boxed{}."
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
        metric=[latex_gold_metric_avg_8],
        generation_size=8192,
        stop_sequence=None,
        few_shots_select="random_sampling_from_train",
    )
]
TASKS_TABLE.extend(gsm8k_tasks)

def get_mgsm_prompt(line, task_name):
    # prompt = polymath_prompt[lang]
    prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    return Doc(
        task_name=task_name, 
        query=f"{line['question']}\n{prompt}",
        choices=[str(line['answer_number'])], 
        gold_index=0
    )

# lang2prompt = {
#     Language.ENGLISH.value: partial(get_mgsm_prompt, lang="en"),
#     Language.FRENCH.value: partial(get_mgsm_prompt, lang="fr"),
#     Language.GERMAN.value: partial(get_mgsm_prompt, lang="de"),
#     Language.RUSSIAN.value: partial(get_mgsm_prompt, lang="ru"),
#     Language.CHINESE.value: partial(get_mgsm_prompt, lang="zh"),
#     Language.JAPANESE.value: partial(get_mgsm_prompt, lang="ja"),
#     Language.THAI.value: partial(get_mgsm_prompt, lang="th"),
#     Language.BENGALI.value: partial(get_mgsm_prompt, lang="bn"),
#     Language.SPANISH.value: partial(get_mgsm_prompt, lang="es"),
# }

# mgsm tasks
mgsm_tasks = [
    LightevalTaskConfig(
        name=f"mgsm:{language.value}",
        prompt_function=get_mgsm_prompt,
        suite=("custom",),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        hf_avail_splits=["train", "test"],
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=[latex_gold_metric_avg_8],
        generation_size=8192,
        stop_sequence=None,
        few_shots_select="random_sampling_from_train",
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
    # lang = line['id'].split("-")[1]
    # prompt = polymath_prompt[lang]
    prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    return Doc(
        task_name=task_name, 
        query=f"{line['question']}\n{prompt}",
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
        metric=[latex_gold_metric_avg_8],
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
