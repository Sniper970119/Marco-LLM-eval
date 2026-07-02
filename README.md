# Marco-LLM Evaluation Scripts

Evaluation scripts for the **Marco-LLM** series of multilingual large language models.  We use the [LightEval](https://github.com/huggingface/lighteval/) library to benchmark our models across a wide range of multilingual and English benchmarks.

## Supported Benchmarks

### Base Model Benchmarks (`base_tasks/`)

| Benchmark | Description |
|-----------|-------------|
| **General Knowledge** | English general knowledge (ARC, HellaSwag, WinoGrande …) |
| **Belebele** | Multilingual reading comprehension (122 languages) |
| **Global PIQA** | Multilingual physical intuition QA |
| **FLORES-200** | Machine translation evaluation (200 languages) |
| **WMT24** | WMT 2024 translation benchmark |
| **INCLUDE** | Multilingual knowledge QA |
| **MGSM** | Multilingual grade-school math |
| **PolyMATH** | Multilingual mathematical reasoning |
| **MLMM HellaSwag** | Multilingual commonsense (HellaSwag) |
| **MLMM ARC** | Multilingual science QA (ARC) |
| **OpenAI MMLU** | Massive multitask language understanding |
| **Global MMLU** | Multilingual MMLU |
| **SIB-200** | Sentence classification (200 languages) |
| **Indo Tasks** | Indonesian NLP benchmarks |
| **M3Exam** | Multilingual multi-level exam QA |
| **MMLU-Pro-X** | Multilingual MMLU-Pro |
| **GPQA** | Graduate-level science QA |
| **Super-GPQA** | Harder GPQA variant |
| **BBH** | BIG-Bench Hard |
| **MATH** | Mathematical problem solving |
| **TydiQA** | Typologically diverse QA |

## Setup

Use conda/uv/venv with `python>=3.11`.

For reproducibility, we recommend fixed versions of the libraries:

```sh
pip install uv
uv venv eval_venv --python 3.11
source eval_venv/bin/activate

GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements.txt
```

## Running the evaluations

The commands below match the configuration used in our evaluation runs.  GPU count is detected automatically.

### Base model evaluation

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # Required for vLLM

MODEL="ATH-MaaS/Marco-Nano-Base"

NUM_GPUS=$([ -n "$CUDA_VISIBLE_DEVICES" ] && echo "$CUDA_VISIBLE_DEVICES" | tr -d ' ' | tr ',' '\n' | grep -v '^$' | wc -l || nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,generation_parameters={temperature:0},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.9"

# Run all base benchmarks at once
bash scripts/eval_base.sh
```

Or run a single benchmark directly:

```bash
lighteval vllm "$MODEL_ARGS" "base_tasks/general_knowledge.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir "evals/$MODEL" \
    --save-details
```

> Note: reasoning-heavy tasks (GPQA, BBH, MATH, PolyMATH, TydiQA) use `max_model_length=8192` and `gpu_memory_utilization=0.6` in the full script.

### Translation evaluation (FLORES+)

```sh
MODEL="ATH-MaaS/Marco-Nano-Base"
python scripts/eval_flores.py --model_path "$MODEL" --base True
```

### Translation evaluation (WMT24++)

```sh
MODEL="ATH-MaaS/Marco-Nano-Base"
python scripts/eval_wmt24.py --model_path "$MODEL" --base True
```

### Belebele evaluation

Belebele is a multilingual reading comprehension benchmark. We evaluate on **29 languages** using multiple-choice format (`mcf`), zero-shot.

Languages covered: Chinese (Simplified), Arabic, German, Spanish, French, Korean, Japanese, Portuguese, Turkish, Indonesian, Italian, Dutch, Polish, Russian, Vietnamese, Thai, Bengali, Czech, Hebrew, Ukrainian, Malay, Urdu, Kazakh, Greek, Romanian, Hungarian, English, Azerbaijani, Nepali.

```bash
MODEL="ATH-MaaS/Marco-Nano-Base"
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,generation_parameters={temperature:0},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.9"

lighteval vllm "$MODEL_ARGS" "base_tasks/belebele.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir "evals/$MODEL" \
    --save-details
```
