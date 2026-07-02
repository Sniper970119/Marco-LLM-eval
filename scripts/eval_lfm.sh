export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

pip install -r requirements.txt

# LFM-MoE (requires specific lighteval version)
pip install scripts/lighteval
pip install vllm==v0.14.0
pip install transformers==5.0.0
pip uninstall -y flash-attn

MODEL="Marco-LLM-Path"

echo "evaluating $MODEL"
NUM_GPUS=8
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_num_batched_tokens=16384,max_model_length=16384,generation_parameters={temperature:1,top_p:1},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.9"
OUTPUT_DIR=evals/$MODEL

# Evaluate English general knowledge
lighteval vllm "$MODEL_ARGS" "it_tasks/general_knowledge.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate global piqa
lighteval vllm "$MODEL_ARGS" "it_tasks/global_piqa.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate include
lighteval vllm "$MODEL_ARGS" "it_tasks/include.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mgsm
lighteval vllm "$MODEL_ARGS" "it_tasks/mgsm.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate polymath
lighteval vllm "$MODEL_ARGS" "it_tasks/polymath.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate mmlu-pro-x
lighteval vllm "$MODEL_ARGS" "it_tasks/mmlu_pro_x.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate gpqa-diamond mgpqa
lighteval vllm "$MODEL_ARGS" "it_tasks/gpqa.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# flores
python scripts/eval_flores.py --model_path "$MODEL" --num_samples 8
# wmt24pp
python scripts/eval_wmt24.py --model_path "$MODEL" --num_samples 8

# Evaluate openai_mmlu
lighteval vllm "$MODEL_ARGS" "it_tasks/openai_mmlu.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

lighteval vllm "$MODEL_ARGS" "it_tasks/global_mmlu_part1.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

lighteval vllm "$MODEL_ARGS" "it_tasks/global_mmlu_part2.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

lighteval vllm "$MODEL_ARGS" "it_tasks/global_mmlu_part3.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details

# Evaluate math
lighteval vllm "$MODEL_ARGS" "it_tasks/math.txt" \
    --use-chat-template \
    --custom-tasks "it_tasks.py" \
    --output-dir $OUTPUT_DIR \
    --save-details
