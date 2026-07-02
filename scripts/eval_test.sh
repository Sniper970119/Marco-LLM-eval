export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

pip install -r requirements.txt

MODEL="Marco-LLM-Path"
MODEL_NAME="Marco-LLM"

echo "evaluating $MODEL"
NUM_GPUS=8
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_num_batched_tokens=8192,max_model_length=8192,generation_parameters={temperature:1,top_p:1},tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.9"
OUTPUT_DIR=evals/$MODEL_NAME

# The loop runs as long as the path does NOT exist
while [ ! -e "$MODEL" ]; do
    echo "Path not found. Checking again in 30 minutes..."
    sleep 30m
done

echo "Path found! Resuming script execution."

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

# Evaluate gpqa-diamond & mgpqa
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

# Evaluate belebele
lighteval vllm "$MODEL_ARGS" "it_tasks/belebele.txt" \
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

# python3 -c "import nltk; nltk.download('punkt_tab')"
# lighteval vllm "$MODEL_ARGS" "extended|ifeval|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR \
#     --save-details

# cd scripts/Multi-IF
# pip install -r requirements.txt
# git clone https://huggingface.co/datasets/facebook/Multi-IF data/Multi-IF
# python multi_turn_instruct_following_eval_vllm.py \
#         --model_path $MODEL \
#         --tokenizer_path $MODEL \
#         --input_data_csv data/Multi-IF/multiIF_20241018.csv \
#         --batch_size 5000
