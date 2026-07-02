export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

MODEL="Marco-LLM-Path"

echo "evaluating $MODEL"

NUM_GPUS=$([ -n "$CUDA_VISIBLE_DEVICES" ] && echo "$CUDA_VISIBLE_DEVICES" | tr -d ' ' | tr ',' '\n' | grep -v '^$' | wc -l || nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected ${NUM_GPUS} GPUs"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,generation_parameters={temperature:0},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.9"
OUTPUT_DIR=evals/$MODEL

pip install -r requirements.txt

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

# Evaluate flores200
python scripts/eval_flores.py --model_path "$MODEL" --base True
python scripts/eval_wmt24.py --model_path "$MODEL" --base True

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

# Evaluate polymath
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=8192,generation_parameters={temperature:0},data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.6"
lighteval vllm "$MODEL_ARGS" "base_tasks/polymath.txt" \
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

# Evaluate sib200
lighteval vllm "$MODEL_ARGS" "base_tasks/sib200.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate global_mmlu
lighteval vllm "$MODEL_ARGS" "base_tasks/global_mmlu.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate indo tasks
lighteval vllm "$MODEL_ARGS" "base_tasks/indo.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate m3exams tasks
lighteval vllm "$MODEL_ARGS" "base_tasks/m3exam.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mmlu_pro_x
lighteval vllm "$MODEL_ARGS" "base_tasks/mmlu_pro_x.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate gpqa
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=8192,generation_parameters={temperature:0},data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.6"
lighteval vllm "$MODEL_ARGS" "base_tasks/gpqa.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate super gpqa
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=8192,generation_parameters={temperature:0},data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.6"
lighteval vllm "$MODEL_ARGS" "base_tasks/super_gpqa.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate bbh
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_num_batched_tokens=8192,max_model_length=8192,generation_parameters={temperature:0},data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.6"
lighteval vllm "$MODEL_ARGS" "base_tasks/bbh.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate MATH
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=8192,generation_parameters={temperature:0},data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.6"
lighteval vllm "$MODEL_ARGS" "base_tasks/math.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate tydiqa
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=8192,generation_parameters={temperature:0},data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.6"
lighteval vllm "$MODEL_ARGS" "base_tasks/tydiqa.txt" \
    --custom-tasks "base_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details
