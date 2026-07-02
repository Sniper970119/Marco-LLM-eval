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

### Instruct Model Benchmarks (`it_tasks/`)

All of the above, plus instruction-following specific variants with chat template support and optional `/think` / `/no_think` control tokens for hybrid-thinking models.

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

All commands below were run on 2 x H100s with 80GB of memory each, using the `vllm` backend.

### Base model evaluation

```bash
MODEL="ATH-MaaS/Marco-Nano-Instruct"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,max_num_batched_tokens=32768,generation_parameters={temperature:0},tensor_parallel_size=2,gpu_memory_utilization=0.7"
lighteval vllm \
    "$MODEL_ARGS" \
    "base_tasks/general_knowledge.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir "evals/" \
    --save-details
```

### Instruct model evaluation (pure reasoning, no hybrid thinking)

```sh 
MODEL="ATH-MaaS/Marco-Nano-Instruct"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=2,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
lighteval vllm "$MODEL_ARGS" "it_tasks/general_knowledge.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir "evals/" \
    --save-details
```

### Instruct model evaluation (with thinking control)

```sh
# Use /think or /no_think to enable or disable extended thinking
SYSTEM_PROMPT="/no_think" 
MODEL="ATH-MaaS/Marco-Nano-Instruct"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=2,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
lighteval vllm "$MODEL_ARGS" "it_tasks/general_knowledge.txt" \
    --use-chat-template \
    --system-prompt "$SYSTEM_PROMPT" \
    --custom-tasks "it_tasks.py" \
    --output-dir "evals/" \
    --save-details
```

### Translation evaluation (FLORES+)

```sh
MODEL="ATH-MaaS/Marco-Nano-Instruct"
python scripts/eval_flores.py --model_path "$MODEL" --num_samples 8
```

### Translation evaluation (WMT24++)

```sh
MODEL="ATH-MaaS/Marco-Nano-Instruct"
python scripts/eval_wmt24.py --model_path "$MODEL" --num_samples 8
```
