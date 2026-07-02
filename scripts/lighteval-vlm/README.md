# LightEval VLM

This directory should contain the LightEval library variant for Vision-Language Models (VLM) evaluation.

## Setup

This is a modified version of [LightEval](https://github.com/huggingface/lighteval/) with VLM support.

Clone and install:

```bash
git clone https://github.com/huggingface/lighteval.git scripts/lighteval-vlm
pip install scripts/lighteval-vlm
```

**Note**: VLM evaluation requires specific versions:

```bash
pip install vllm==v0.14.0
pip install transformers==5.0.0
pip uninstall -y flash-attn
```
