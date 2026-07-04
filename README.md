# LLM Fine-Tuning

Fine-tuning large language models using LoRA and QLoRA. Each project covers the full pipeline: dataset preparation, adapter training, evaluation, and deployment to Hugging Face Hub.

## Projects

| Project | Model | Method | Dataset | HF Hub |
|---|---|---|---|---|
| [Code Generator](./code_generator) | Qwen2.5-0.5B | LoRA | Vezora/Tested-22k-Python-Alpaca | [link](https://huggingface.co/Akram321/qwen2.5-code-generator-lora) |
| Arabic LLM (coming soon) | | | |  |

## Stack

- [PEFT](https://github.com/huggingface/peft) LoRA adapter injection
- [TRL](https://github.com/huggingface/trl) SFT training loop
- [Unsloth](https://github.com/unslothai/unsloth) QLoRA with 4-bit quantization
- [Hugging Face Hub](https://huggingface.co/Akram321) model hosting
