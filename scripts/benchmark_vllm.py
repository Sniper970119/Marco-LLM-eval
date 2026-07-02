import time
import argparse
from vllm import LLM, SamplingParams


def run_benchmark(model_name, prompts):
    """Benchmark vLLM throughput on a single GPU."""
    llm = LLM(model=model_name, max_model_len=8192)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    # Warm-up run to JIT compile and load weights into cache
    llm.generate(["Warm up"], sampling_params)

    for batch_size in range(1000, 11000, 1000):
        print(f"--- Starting Benchmark for {model_name} with batch size {batch_size}---")
        start_time = time.perf_counter()
        outputs = llm.generate(prompts * batch_size, sampling_params)
        end_time = time.perf_counter()

        total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        duration = end_time - start_time
        throughput = total_generated_tokens / duration

        print(f"Results for {model_name}:")
        print(f"  Total Tokens: {total_generated_tokens}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Throughput: {throughput:.2f} tokens/sec\n")


def compute_bleu(prediction_file):
    """Compute BLEU score from a predictions JSONL file."""
    import json
    import sacrebleu

    metric = sacrebleu.BLEU(tokenize="flores200")
    targets, predictions = [], []
    with open(prediction_file) as f:
        for jsonline in f.readlines():
            example = json.loads(jsonline)
            targets.append(example['target'])
            predictions.append(example['predictions'][0])

    for i in range(len(targets)):
        print(targets[i] + "\t" + predictions[i])
    print(float(metric.corpus_score(hypotheses=predictions, references=[targets]).score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Marco-LLM-Path", help="Path to the model")
    parser.add_argument("--prediction_file", type=str, default=None, help="Path to predictions JSONL for BLEU scoring")
    args = parser.parse_args()

    if args.prediction_file:
        compute_bleu(args.prediction_file)
    else:
        test_prompts = ["Explain the theory of relativity in simple terms."]
        run_benchmark(args.model_path, test_prompts)