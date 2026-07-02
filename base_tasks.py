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
from functools import partial
import numpy as np
import json

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
    SampleLevelMetricGrouping,
)
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

import nltk
nltk.download('punkt_tab')

TASKS_TABLE = []
TASKS_TABLE.extend(ML_TASKS_TABLE)

#------------------
# BASE MODEL EVALS
#------------------
qa_metrics = [
    loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
    loglikelihood_acc_metric(normalization=LogProbCharNorm()),
]
all_qa_formulations = [MCFFormulation(), CFFormulation(), HybridFormulation()]

# ARC tasks
arc_tasks = [
    LightevalTaskConfig(
        name=f"arc_{formulation.name.lower()}:{subset.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="allenai/ai2_arc",
        hf_subset=f"ARC-{subset}",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in ["Easy", "Challenge"]
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(arc_tasks)

# multilingual ARC challenge tasks
mlmm_arc_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_arc_{lang.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(lang.value),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="train",
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
        Language.NEPALI,
        Language.DANISH,
        Language.SWEDISH,
        Language.CATALAN,
        Language.GUJARATI,
        Language.HINDI,
        Language.CROATIAN,
        Language.MARATHI,
        Language.SLOVAK,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.TAMIL,
        Language.TELUGU,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]
TASKS_TABLE.extend(mlmm_arc_tasks)

# BoolQ task
boolq_task = LightevalTaskConfig(
    name="boolq_cf",
    prompt_function=get_boolq_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "answer": line["answer"],
            "context": line["passage"],
        },
        formulation=CFFormulation(),
    ),
    suite=("custom",),
    hf_repo="google/boolq",
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    generation_size=5,
    stop_sequence=["\n"],
    metric=get_metrics_for_formulation(CFFormulation(), qa_metrics),
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
        Language.NEPALI,
        Language.DANISH,
        Language.SWEDISH,
        Language.CATALAN,
        Language.GUJARATI,
        Language.HINDI,
        Language.CROATIAN,
        Language.MARATHI,
        Language.SLOVAK,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.TAMIL,
        Language.TELUGU,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]
TASKS_TABLE.extend(mlmm_hellaswag_tasks)

# CommonsenseQA tasks
commonsense_qa_tasks = [
    LightevalTaskConfig(
        name=f"commonsenseqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": line["choices"]["label"].index(line["answerKey"].strip()),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_revision="94630fe30dad47192a8546eb75f094926d47e155",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(commonsense_qa_tasks)

# OpenBookQA tasks
openbook_qa_tasks = [
    LightevalTaskConfig(
        name=f"openbookqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question_stem"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
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
        few_shots_split="train",
        metric=qa_metrics,
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(winogrande_tasks)

# PIQA tasks
piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["goal"],
                "choices": [line["sol1"], line["sol2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        hf_subset="plain_text",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(piqa_tasks)

# Global PIQA tasks
global_piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_{language}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["prompt"],
                "choices": [line["solution0"], line["solution1"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="mrlbenchmarks/global-piqa-nonparallel",
        hf_subset=language,
        trust_dataset=True,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
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

# SIQA tasks
siqa_tasks = [
    LightevalTaskConfig(
        name=f"siqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [line["answerA"], line["answerB"], line["answerC"]],
                "gold_idx": int(line["label"]) - 1,
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="allenai/social_i_qa",
        hf_revision="53620e5841fb12b08e082485797e7021d3684ea2",
        hf_subset="default",
        trust_dataset=True,
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
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
# fmt: on

mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="cais/mmlu",
        hf_subset=subset,
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in MMLU_SUBSETS
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_tasks)

mmlu_redux_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_redux_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_subset=subset,
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="test",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in MMLU_SUBSETS
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_redux_tasks)

agi_eval_tasks = [
    LightevalTaskConfig(
        name=f"agi_eval_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": [option[3:] for option in line["options"]],
                "gold_idx": LETTER_INDICES.index(line["label"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="lighteval/agi_eval_en",
        hf_subset=subset,
        trust_dataset=True,
        evaluation_splits=("train",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(agi_eval_tasks)

# MMLU Pro tasks
mmlu_pro_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_pro_tasks)

def get_mmlu_pro_cot_prompt_function(line, task_name):
    MMLU_Pro_QUERY_TEMPLATE = """
Answer the following multiple choice question. Think step by step before answering.

Question: {Question}

""".strip()
    if line["cot_content"] != '':
        gold = f" {line['cot_content'][len('A:') + 1 :]}"
    else:
        gold = f"({line['answer']})"

    ids2choice = {
        0: '(A)',
        1: '(B)',
        2: '(C)',
        3: '(D)',
        4: '(E)',
        5: '(F)',
        6: '(G)',
        7: '(H)',
        8: '(I)',
        9: '(J)',
    }
    query = MMLU_Pro_QUERY_TEMPLATE.format(Question=line["question"]) + "\n"
    if "options" not in line:
        line["options"] = [line[f"option_{i}"] for i in range(10)]
    for i in range(len(line["options"])):
        query += f"{ids2choice[i]}: {line['options'][i]}\n"
    
    query += "A:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[gold],
        gold_index=0,
    )

# MMLU Pro CoT tasks
mmlu_pro_cot_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_cot",
        prompt_function=get_mmlu_pro_cot_prompt_function,
        suite=("custom",),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        generation_size=512,
        metric=[Metrics.gpqa_instruct_metric],
        # stop_sequence=['\n'],
        stop_sequence=None,
        version=0,
    )
]
TASKS_TABLE.extend(mmlu_pro_cot_tasks)

# MMLU Pro X tasks
mmlu_pro_x_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_x_lite_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": [line[f"option_{i}"] for i in range(10)],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="li-lab/MMLU-ProX-Lite",
        hf_subset=standardize_tag(language.value),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for language in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CZECH,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.HINDI,
        Language.HUNGARIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.MARATHI,
        Language.NEPALI,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SERBIAN,
        Language.SWAHILI,
        Language.TELUGU,
        Language.THAI,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
        Language.YORUBA,
        Language.CHINESE,
        Language.ZULU,
    ]
    for formulation in all_qa_formulations
]
TASKS_TABLE.extend(mmlu_pro_x_tasks)

# MMLU Pro X CoT tasks
mmlu_pro_x_cot_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_x_cot_{language.value}",
        prompt_function=get_mmlu_pro_cot_prompt_function,
        suite=("custom",),
        hf_repo="li-lab/MMLU-ProX",
        hf_subset=standardize_tag(language.value),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        generation_size=512,
        metric=[Metrics.gpqa_instruct_metric],
        # stop_sequence=['\n'],
        stop_sequence=None,
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
TASKS_TABLE.extend(mmlu_pro_x_cot_tasks)

def get_gpqa_cot_prompt_function(line, task_name):
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

    with open("prompts/gpqa.json") as f:
        gpqa_nshot_prompt = json.load(f)

    INSTRUCTION = """Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand."""
    GPQA_QUERY_TEMPLATE = """Question: {Question}
Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}\nLet's think step by step: """
    
    choices = [
        preprocess(line["Incorrect Answer 1"]),
        preprocess(line["Incorrect Answer 2"]),
        preprocess(line["Incorrect Answer 3"]),
        preprocess(line["Correct Answer"]),
    ]

    random.shuffle(choices)
    correct_answer_index = choices.index(preprocess(line["Correct Answer"]))

    out_doc = {
        "choice1": choices[0],
        "choice2": choices[1],
        "choice3": choices[2],
        "choice4": choices[3],
    }
    query = GPQA_QUERY_TEMPLATE.format(
        Question=line["Question"],
        **out_doc,
    )
    # target = line["Explanation"] + f" The correct answer is ({chr(65 + correct_answer_index)})."
    target = chr(65 + correct_answer_index)
    return Doc(
        task_name=task_name,
        query=INSTRUCTION + "\n" + "\n\n".join(gpqa_nshot_prompt + [query]),
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
        few_shots_split="train",
        generation_size=4096,
        metric=[Metrics.gpqa_instruct_metric],
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
        few_shots_split="train",
        generation_size=4096,
        metric=[Metrics.gpqa_instruct_metric],
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
    with open("prompts/super_gpqa.json") as f:
        super_gpqa_nshot_prompt = json.load(f)

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

    INSTRUCTION = """Answer the following multiple-choice question. There is only one correct answer. The last line of your response should be in the format Answer: $LETTER (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J."""
    GPQA_QUERY_TEMPLATE = """Question: {Question} 
"""
    
    query = GPQA_QUERY_TEMPLATE.format(Question=line["question"])
    for i in range(len(line['options'])):
        query += ids2choice[i] + " " + line['options'][i] + "\n"
    query += "Answer: Let's think step by step:"
    return Doc(
        task_name=task_name,
        query="\n\n".join([INSTRUCTION] + super_gpqa_nshot_prompt + [query]),
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
        few_shots_split="train",
        generation_size=4096,
        metric=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Question:"],
        version=0,
    )
]
TASKS_TABLE.extend(supergpqa_cot_tasks)

# BELEBELE tasks
belebele_tasks = [
    LightevalTaskConfig(
        name=f"belebele_{language}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            # iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language if language not in ['heb_Hebr', 'ben_Beng', 'kaz_Cyrl', 'kor_Hang'] else 'eng_Latn').to_alpha3()],
            iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get('eng_Latn').to_alpha3()],
            lambda line: {
                "question": line["question"],
                "context": line["flores_passage"],
                "choices": [line[f"mc_answer{i}"] for i in range(1, 5)],
                "gold_idx": int(line["correct_answer_num"]) - 1,
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="facebook/belebele",
        hf_subset=language,
        evaluation_splits=("test",),
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
        "azj_Latn",
        "npi_Deva",
        "amh_Ethi",
        "cat_Latn",
        "est_Latn",
        "tgl_Latn",
        "guj_Gujr",
        "hrv_Latn",
        "jav_Latn",
        "lao_Laoo",
        "lvs_Latn",
        "mlt_Latn",
        "nob_Latn",
        "slk_Latn",
        "srp_Cyrl",
        "tam_Taml",
        "yor_Latn",
        "bul_Cyrl",
        "eus_Latn",
        "fin_Latn",
        "hin_Deva",
        "ibo_Latn",
        "khm_Khmr",
        "lit_Latn",
        "mar_Deva",
        "mya_Mymr",
        "pan_Guru",
        "slv_Latn",
        "swh_Latn",
        "tel_Telu",
        "zul_Latn",
        "dan_Latn",
        "swe_Latn",
        "pes_Arab",
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

# INCLUDE tasks
include_tasks = [
    LightevalTaskConfig(
        name=f"include_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            # language if language not in [Language.HEBREW, Language.BENGALI, Language.KAZAKH, Language.KOREAN] else Language.ENGLISH,
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": [line[f"option_{c}"] for c in ['a', 'b', 'c', 'd']],
                "gold_idx": int(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="CohereLabs/include-base-44",
        hf_subset=str(language).split(".")[1].capitalize(),
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        few_shots_split="validation",
        # hf_filter=partial(
        #     lambda region_feature, line: line['regional_feature'] == region_feature,
        #     region_feature,
        # ),
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
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
        Language.BASQUE,
        Language.CROATIAN,
        Language.LITHUANIAN,
        Language.ESTONIAN,
        Language.FINNISH,
        Language.SERBIAN,
        Language.BULGARIAN,
        Language.PERSIAN,
        Language.HINDI,
        Language.TAMIL,
        Language.TELUGU,
        Language.TAGALOG,
    ]
    # for region_feature in [
    #     'region implicit',
    #     'region explicit',
    #     'culture',
    #     'agnostic',
    # ]
]
TASKS_TABLE.extend(include_tasks)

# global_mmlu tasks
global_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"global_mmlu_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            # language if language not in [Language.HEBREW, Language.BENGALI, Language.KOREAN] else Language.ENGLISH,
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": [line["option_a"], line["option_b"], line["option_c"], line["option_d"]],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
        Language.TAGALOG,
        Language.HINDI,
        Language.IGBO,
        Language.LITHUANIAN,
        Language.PERSIAN,
        Language.SERBIAN,
        Language.SWAHILI,
        Language.SWEDISH,
        Language.TELUGU,
        Language.YORUBA,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(global_mmlu_tasks)

# -------------------- Language Specific MMLU tasks --------------------
cmmlu_tasks = [
    LightevalTaskConfig(
        name=f"zho_mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["Question"],
                "choices": [line["A"], line["B"], line["C"], line["D"]],
                "gold_idx": LETTER_INDICES.index(line["Answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="lmlmcat/cmmlu",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(cmmlu_tasks)

ceval_tasks = [
    LightevalTaskConfig(
        name=f"ceval_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["question"],
                "choices": [line["A"], line["B"], line["C"], line["D"]],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="ceval/ceval-exam",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(ceval_tasks)

# arabic mmlu tasks
arabic_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"ara_mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ARABIC,
            lambda line: {
                "context": line["Context"],
                "question": line["Question"],
                "choices": [str(o) for o in [line[f"Option {i}"] for i in range(1, 6)] if o],
                "gold_idx": LETTER_INDICES.index(line["Answer Key"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="MBZUAI/ArabicMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        hf_avail_splits=["dev"],
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(arabic_mmlu_tasks)

# turkish mmlu tasks
turkish_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"tur_mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="AYueksel/TurkishMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(turkish_mmlu_tasks)

# KazMMLU tasks
kaz_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"kaz_mmlu_{formulation.name.lower()}:{subject}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["Question"],
                "choices": [line["Option A"], line["Option B"], line["Option C"], line["Option D"], line["Option E"]],
                "gold_idx": LETTER_INDICES.index(line["Answer Key"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="MBZUAI/KazMMLU",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(kaz_mmlu_tasks)

# GreekMMLU tasks
greek_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"greek_mmlu_{formulation.name.lower()}:{subject}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": line["answer"],
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="dascim/GreekMMLU",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(greek_mmlu_tasks)

# IndoCareer tasks
indo_career_tasks = [
    LightevalTaskConfig(
        name=f"indo_career_{formulation.name.lower()}:{subject}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "instruction": f"Ini adalah soal {subject} untuk {line['Exam Type']}. Pilihlah salah satu jawaban yang dianggap benar!",
                "question": line["Question"],
                "choices": [line["Option A"], line["Option B"], line["Option C"], line["Option D"], line["Option E"]],
                "gold_idx": LETTER_INDICES.index(line["Answer Key"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="indolem/IndoCareer",
        hf_subset=subject,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(indo_career_tasks)

# IndoCulture tasks
indo_culture_tasks = [
    LightevalTaskConfig(
        name=f"indo_culture_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": f"Untuk konteks {line['province']}, sambungan yang tepat dari kalimat {line['context']} adalah",
                "choices": [option[3:] for option in eval(line["options"])],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="indolem/IndoCulture",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split=None,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
TASKS_TABLE.extend(indo_culture_tasks)

# IndoMMLU tasks
indo_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"indo_mmlu_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "instruction": f"Ini adalah soal {line['subject']} untuk {line['level']}. Pilihlah salah satu jawaban yang dianggap benar!",
                "question": line['question'],
                "choices": [option[3:] for option in eval(line["options"])],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="indolem/IndoMMLU",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split=None,
        trust_dataset=True,
        metric=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
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
openai_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"openai_mmlu_{language[0].value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["Question"],
                "choices": [line["A"], line["B"], line["C"], line["D"]],
                "gold_idx": LETTER_INDICES.index(line["Answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="openai/MMMLU",
        hf_subset=language[1],
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        hf_revision="038c7808122969ead7456361af05cb8f47d247f8",
        metric=get_metrics_for_formulation(formulation, qa_metrics),
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
        (Language.HINDI, "HI_IN"),
        (Language.SWAHILI, "SW_KE"),
        (Language.YORUBA, "YO_NG"),
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
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
        generation_size=400,
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

lang2prompt = {
    Language.ENGLISH.value: prompt.mgsm_en,
    Language.FRENCH.value: prompt.mgsm_fr,
    Language.GERMAN.value: prompt.mgsm_de,
    Language.RUSSIAN.value: prompt.mgsm_ru,
    Language.CHINESE.value: prompt.mgsm_zh,
    Language.JAPANESE.value: prompt.mgsm_ja,
    Language.THAI.value: prompt.mgsm_th,
    Language.BENGALI.value: prompt.mgsm_bn,
    Language.SPANISH.value: prompt.mgsm_es,
    Language.SWAHILI.value: prompt.mgsm_sw,
    Language.TELUGU.value: prompt.mgsm_te,
}
stop_symbol = {
    Language.ENGLISH.value: ["Question:"],
    Language.FRENCH.value: ["Question :", "Question:"],
    Language.GERMAN.value: ["Frage:"],
    Language.RUSSIAN.value: ["Задача:"],
    Language.CHINESE.value: ["问题：", "问题:"],
    Language.JAPANESE.value: ["問題：", "問題:"],
    Language.THAI.value: ["โจทย์:"],
    Language.BENGALI.value: ["প্রশ্ন:"],
    Language.SPANISH.value: ["Pregunta:"],
    Language.SWAHILI.value: ["Swali:"],
    Language.TELUGU.value: ["ప్రశ్న:"],
}
# mgsm tasks
mgsm_tasks = [
    LightevalTaskConfig(
        name=f"mgsm:{language.value}",
        prompt_function=lang2prompt[language.value],
        suite=("custom",),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        hf_avail_splits=["train", "test"],
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=[Metrics.exact_match, Metrics.quasi_exact_match, Metrics.expr_gold_metric],
        generation_size=512,
        # stop_sequence=['\n', '\n\n'],
        stop_sequence=stop_symbol[language.value],
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
        Language.SWAHILI,
        Language.TELUGU,
    ]
]
TASKS_TABLE.extend(mgsm_tasks)

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
        generation_size=300,
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
        generation_size=256,
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
        generation_size=4096,
        metric=[Metrics.gpqa_instruct_metric],
        stop_sequence=["Q:"],
    )
    for subset in BBH_SUBSETS
]
TASKS_TABLE.extend(bbh_tasks)

# GSM8K tasks
gsm8k_tasks = [
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function=prompt.gsm8k,
        suite=("custom",),
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.expr_gold_metric],
        generation_size=256,
        stop_sequence=["Question:"],
        few_shots_select="random_sampling_from_train",
    )
]
TASKS_TABLE.extend(gsm8k_tasks)

# MATH tasks
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

math_tasks = [
    LightevalTaskConfig(
        name=f"math_cot:{config}",
        suite=("custom",),
        prompt_function=prompt.math_cot,
        hf_repo="DigitalLearningGmbH/MATH-lighteval",
        hf_subset=config,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=4096,
        metric=[latex_gold_metric],
        stop_sequence=["\n"],
        # stop_sequence=None,
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

def get_polymath_prompt(line, task_name: str = None):
    lang = line['id'].split("-")[1]
    prompt = polymath_prompt[lang]
    return Doc(
        task_name=task_name, 
        query=f"{line['question']}\n{prompt}", 
        # query=f"{line['question']}",
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
        metric=[latex_gold_metric],
        generation_size=4096,
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
