export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

# Requires specific versions for Trinity models
# pip install vllm==v0.14.0
# pip install transformers==5.0.0
# pip uninstall -y flash-attn

NUM_GPUS=8

MODEL="Marco-LLM-Path"

# Base model evaluation
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,trust_remote_code=true,max_num_batched_tokens=4096,max_model_length=4096,generation_parameters={temperature:0},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.9"
OUTPUT_DIR=evals/$MODEL

# Evaluate English general knowledge
lighteval vllm "$MODEL_ARGS" "base_tasks/general_knowledge.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate belebele
lighteval vllm "$MODEL_ARGS" "base_tasks/belebele.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate global piqa
lighteval vllm "$MODEL_ARGS" "base_tasks/global_piqa.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate include
lighteval vllm "$MODEL_ARGS" "base_tasks/include.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mgsm
lighteval vllm "$MODEL_ARGS" "base_tasks/mgsm.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mlmm hellaswag
lighteval vllm "$MODEL_ARGS" "base_tasks/mlmm_hellaswag.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mlmm arc
lighteval vllm "$MODEL_ARGS" "base_tasks/mlmm_arc.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate openai_mmlu
lighteval vllm "$MODEL_ARGS" "base_tasks/openai_mmlu.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,trust_remote_code=true,max_num_batched_tokens=8192,max_model_length=8192,generation_parameters={temperature:0},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.6"
lighteval vllm "$MODEL_ARGS" "base_tasks/mmlu_pro_x.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate bbh
lighteval vllm "$MODEL_ARGS" "base_tasks/bbh.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

lighteval vllm "$MODEL_ARGS" "base_tasks/global_mmlu.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# flores
python scripts/eval_flores.py --model_path "$MODEL" --base True
python scripts/eval_wmt24.py --model_path "$MODEL" --base True
