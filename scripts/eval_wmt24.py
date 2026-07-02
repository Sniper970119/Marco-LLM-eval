from vllm import LLM, SamplingParams
import pandas as pd
import os
import sacrebleu
from transformers import AutoTokenizer
import argparse
import json
import random
from datasets import load_dataset

random.seed(42)


def main(lang_id, lang, file_name, llm, tokenizer, num_samples, base, is_vlm, add_suffix, en_xx=True):
    lang_prompt = """Please translate the following text from English to {lang}:\n{question}\n"""  if en_xx else """Please translate the following text from {lang} to English:\n{question}\n"""

    dataset = load_dataset("google/wmt24pp", f"en-{lang_id}")
    source_texts = [example["source"] for example in dataset["train"]]
    target_texts = [example["target"] for example in dataset["train"]]

    if not en_xx:
        source_texts, target_texts = target_texts, source_texts

    if base:
        dev_source_dataset = pd.read_parquet("data/flores_plus/dev/eng_Latn.parquet")
        dev_source_texts = dev_source_dataset["text"].tolist()
        dev_target_dataset = pd.read_parquet(os.path.join("data/flores_plus/dev", file_name))
        dev_target_texts = dev_target_dataset["text"].tolist()

        if not en_xx:
            dev_source_texts, dev_target_texts = dev_target_texts, dev_source_texts

        few_shots = [lang_prompt.format(lang=lang, question=src) + tgt for src, tgt in zip(dev_source_texts, dev_target_texts)]
        random.shuffle(few_shots)
        few_shots = "\n\n".join(few_shots[:5])

        prompts = [few_shots + "\n\n" + lang_prompt.format(lang=lang, question=q) for q in source_texts]
    else:
        prompts = [lang_prompt.format(lang=lang, question=q) for q in source_texts]

    # greedy for base models
    if base:
        sampling_params = SamplingParams(temperature=0, max_tokens=512, n=num_samples)
    else:
        sampling_params = SamplingParams(temperature=1., top_p=1., max_tokens=512, n=num_samples)

    if not base:
        if add_suffix:
            prompts = [prompt + "You should only output the translated texts without any other content.\n" for prompt in prompts]
        messages = [
            [{"role": "user", "content": prompt}] for prompt in prompts
        ]
        if is_vlm:
            prompts = []
            for message in messages:
                prompt_dict = {"prompt": tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)}
                prompts.append(prompt_dict)
        else:
            prompts = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

    predictions = [[] for _ in range(num_samples)]
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        for i in range(num_samples):
            text = output.outputs[i].text
            predictions[i].append(text.split("\n\n")[0].split("\n")[0])
    
    metric = sacrebleu.CHRF(word_order=2)
    chrf_score = sum([float(metric.corpus_score(hypotheses=predictions[i], references=[target_texts]).score) for i in range(num_samples)]) / num_samples
    print(f"{lang} CHRF++ score: {chrf_score:.1f}")
    metric = sacrebleu.BLEU(tokenize="flores200")
    bleu_score = sum([float(metric.corpus_score(hypotheses=predictions[i], references=[target_texts]).score) for i in range(num_samples)]) / num_samples
    print(f"{lang} BLEU score: {bleu_score:.1f}")

    output_dir = f"evals/{model_path}/flores_plus/{lang}"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"flores_dev_test_English_{lang}_metrics.json" if en_xx else f"flores_dev_test_{lang}_English_metrics.json"), "w") as f:
        json.dump({
            "chrf++": chrf_score,
            "bleu": bleu_score,
        }, f, indent=4)
    with open(os.path.join(output_dir, f"flores_dev_test_English_{lang}_predictions.jsonl" if en_xx else f"flores_dev_test_{lang}_English_predictions.jsonl"), "w") as f:
        for i in range(len(source_texts)):
            f.write(json.dumps({
                "full_prompt": prompts[i],
                "source": source_texts[i],
                "target": target_texts[i],
                "predictions": [predictions[j][i] for j in range(num_samples)],
            }, ensure_ascii=False))
            f.write("\n")
    
    return chrf_score, bleu_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for vllm generation")
    parser.add_argument("--base", type=bool, default=False, help="Whether to use base models")
    parser.add_argument("--is_vlm", type=bool, default=False, help="Whether to use VLMs")
    parser.add_argument("--add_suffix", type=bool, default=False, help="Whether to add suffix to prompt")
    args = parser.parse_args()
    model_path = args.model_path
    num_samples = args.num_samples
    seed = args.seed
    base = args.base
    is_vlm = args.is_vlm
    add_suffix = args.add_suffix

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        try:
            # Try using nvidia-smi to count GPUs
            import subprocess
            nvidia_smi = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"])
            num_gpus = len(nvidia_smi.decode().strip().split("\n"))
        except:
            print("nvidia-smi not found. Defaulting to 1 GPU")
            num_gpus = 1
    print(f"Detected {num_gpus} GPUs")
    
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, dtype="bfloat16", seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path) 

    en_xx_chrf_score, en_xx_bleu_score = {}, {}
    xx_en_chrf_score, xx_en_bleu_score = {}, {}
    for lang_id, lang, file_name in [
        ("ar_EG", 'Arabic', 'arb_Arab.parquet'),
        ("ar_SA", 'Arabic', 'arb_Arab.parquet'),
        ("bn_IN", 'Bengali', 'ben_Beng.parquet'),
        ("de_DE", 'German', 'deu_Latn.parquet'),
        ("es_MX", 'Spanish', 'spa_Latn.parquet'),
        ("fr_CA", 'French', 'fra_Latn.parquet'),
        ("fr_FR", 'French', 'fra_Latn.parquet'),
        ("id_ID", 'Indonesian', 'ind_Latn.parquet'),
        ("it_IT", 'Italian', 'ita_Latn.parquet'),
        ("ja_JP", 'Japanese', 'jpn_Jpan.parquet'),
        ("ko_KR", 'Korean', 'kor_Hang.parquet'),
        ("pt_BR", 'Portuguese', 'por_Latn.parquet'),
        ("pt_PT", 'Portuguese', 'por_Latn.parquet'),
        ("ru_RU", 'Russian', 'rus_Cyrl.parquet'),
        ("th_TH", 'Thai', 'tha_Thai.parquet'),
        ("vi_VN", 'Vietnamese', 'vie_Latn.parquet'),
        ("zh_CN", 'Chinese', 'cmn_Hans.parquet'),
        ("zh_TW", 'Chinese', 'cmn_Hans.parquet'),
        ("pl_PL", 'Polish', 'pol_Latn.parquet'),
        ("cs_CZ", 'Cezech', 'ces_Latn.parquet'),
        ("uk_UA", 'Ukrainian', 'ukr_Cyrl.parquet'),
        ("hu_HU", 'Hungarian', 'hun_Latn.parquet'),
        ("ro_RO", 'Romanian', 'ron_Latn.parquet'),
        ("he_IL", 'Hebrew', 'heb_Hebr.parquet'),
        ("ur_PK", 'Urdu', 'urd_Arab.parquet'),
        ("el_GR", 'Greek', 'ell_Grek.parquet'),
        ("nl_NL", 'Dutch', 'nld_Latn.parquet'),
        ("tr_TR", 'Turkish', 'tur_Latn.parquet'),
        # ("bg_BG", 'Bulgarian', 'bul_Cyrl.parquet'),
        # ("ca_ES", 'Catalan', 'cat_Latn.parquet'),
        # ("da_DK", 'Danish', 'dan_Latn.parquet'),
        # ("et_EE", 'Estonian', 'ekk_Latn.parquet'),
        # ("fa_IR", 'Persian', 'pes_Arab.parquet'),
        # ("fil_PH", 'Tagalog', 'fil_Latn.parquet'),
        # ("gu_IN", 'Gujarati', 'guj_Gujr.parquet'),
        # ("hi_IN", 'Hindi', 'hin_Deva.parquet'),
        # ("hr_HR", 'Croatian', 'hrv_Latn.parquet'),
        # ("lt_LT", 'Lithuanian', 'lit_Latn.parquet'),
        # ("lv_LV", 'Latvian', 'lvs_Latn.parquet'),
        # ("mr_IN", 'Marathi', 'mar_Deva.parquet'),
        # ("no_NO", 'Norwegian', 'nob_Latn.parquet'),
        # ("pa_IN", 'Punjabi', 'pan_Guru.parquet'),
        # ("sk_SK", 'Slovak', 'slk_Latn.parquet'),
        # ("sl_SI", 'Slovenian', 'slv_Latn.parquet'),
        # ("sr_RS", 'Serbian', 'srp_Cyrl.parquet'),
        # ("sv_SE", 'Swedish', 'swe_Latn.parquet'),
        # ("sw_KE", 'Swahili', 'swh_Latn.parquet'),
        # ("sw_TZ", 'Swahili', 'swh_Latn.parquet'),
        # ("ta_IN", 'Tamil', 'tam_Taml.parquet'),
        # ("te_IN", 'Telugu', 'tel_Telu.parquet'),
        # ("zu_ZA", 'Zulu', 'zul_Latn.parquet')
    ]:
        chrf_score, bleu_score = main(lang_id, lang, file_name, llm, tokenizer, num_samples, base, is_vlm, add_suffix, en_xx=True)
        en_xx_chrf_score[lang_id] = chrf_score
        en_xx_bleu_score[lang_id] = bleu_score
        chrf_score, bleu_score = main(lang_id, lang, file_name, llm, tokenizer, num_samples, base, is_vlm, add_suffix, en_xx=False)
        xx_en_chrf_score[lang_id] = chrf_score
        xx_en_bleu_score[lang_id] = bleu_score
    
    print("=" * 32)
    print("EN-XX CHRF++ score: ")
    print(json.dumps(en_xx_chrf_score, indent=2))
    print("EN-XX BLEU score: ")
    print(json.dumps(en_xx_bleu_score, indent=2))
    print("=" * 32)
    print("XX-EN CHRF++ score: ")
    print(json.dumps(xx_en_chrf_score, indent=2))
    print("XX-EN BLEU score: ")
    print(json.dumps(xx_en_bleu_score, indent=2))

    print(f"EN-XX CHRF++ score: {sum(list(en_xx_chrf_score.values())) / len(en_xx_chrf_score):.1f}")
    print(f"EN-XX BLEU score: {sum(list(en_xx_bleu_score.values())) / len(en_xx_bleu_score):.1f}")
    print(f"XX-EN CHRF++ score: {sum(list(xx_en_chrf_score.values())) / len(xx_en_chrf_score):.1f}")
    print(f"XX-EN BLEU score: {sum(list(xx_en_bleu_score.values())) / len(xx_en_bleu_score):.1f}")