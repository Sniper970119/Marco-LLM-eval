export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

MODEL="Marco-LLM-Path"
MODEL_NAME="Marco-LLM"
echo "evaluating $MODEL"

NUM_GPUS=4
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_num_batched_tokens=32768,max_model_length=8192,generation_parameters={temperature:0.7,top_p:0.8,top_k:20,min_p:0},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.9"
OUTPUT_DIR=evals/$MODEL_NAME

pip install -r requirements.txt

# Evaluate English general knowledge
CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/general_knowledge.txt" \
    --custom-tasks "it_tasks_zy.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate belebele
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/belebele_zy.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate flores200
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/flores200_zy.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks_zy.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate include
lighteval vllm "$MODEL_ARGS" "it_tasks/include.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mgsm
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/mgsm_zy.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks_zy.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate polymath
lighteval vllm "$MODEL_ARGS" "it_tasks/polymath.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mlmm hellaswag
lighteval vllm "$MODEL_ARGS" "it_tasks/mlmm_hellaswag.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mlmm arc
lighteval vllm "$MODEL_ARGS" "it_tasks/mlmm_arc.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate openai_mmlu
lighteval vllm "$MODEL_ARGS" "it_tasks/openai_mmlu.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate sib200
lighteval vllm "$MODEL_ARGS" "it_tasks/sib200.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate global_mmlu (split into 3 parts)
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/global_mmlu_part1.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/global_mmlu_part2.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/global_mmlu_part3.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate indo tasks
lighteval vllm "$MODEL_ARGS" "it_tasks/indo.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate m3exams tasks
lighteval vllm "$MODEL_ARGS" "it_tasks/m3exam.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mmlu_pro_x
lighteval vllm "$MODEL_ARGS" "it_tasks/mmlu_pro_x.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate gpqa
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/gpqa.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate super gpqa
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/super_gpqa.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate bbh
lighteval vllm "$MODEL_ARGS" "it_tasks/bbh.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate MATH
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1,2,3 lighteval vllm "$MODEL_ARGS" "it_tasks/math.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate tydiqa
lighteval vllm "$MODEL_ARGS" "it_tasks/tydiqa.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details
