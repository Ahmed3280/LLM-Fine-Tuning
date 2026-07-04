# LoRA Fine-Tuning: Python Code Generator

Fine-tuning `Qwen/Qwen2.5-0.5B` on 22k Python instruction-response pairs using LoRA. The model learns to follow a structured format and generate Python code with explanations and usage examples.

## Model

Adapter live on HF Hub: [Akram321/qwen2.5-code-generator-lora](https://huggingface.co/Akram321/qwen2.5-code-generator-lora)

Base model: `Qwen/Qwen2.5-0.5B` | Dataset: `Vezora/Tested-22k-Python-Alpaca`

## Stack

- `transformers` model loading and inference
- `peft` LoRA adapter injection
- `trl` SFTTrainer training loop
- `datasets` dataset loading and formatting

## LoRA Config

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Target modules | q_proj, v_proj |
| Trainable parameters | 540,672 (0.11% of total) |
| Dropout | 0.05 |

## Training

| Setting | Value |
|---|---|
| Hardware | Google Colab T4 (15GB) |
| Epochs | 1 |
| Batch size | 4 (effective 16 with grad accumulation) |
| Learning rate | 2e-4 |
| Precision | fp16 |
| Steps | 1,413 |
| Final loss | 0.599 |
| Mean token accuracy | 81.6% |

## Before vs After Fine-Tuning

**Prompt:**
```
### Instruction:
Write a Python function that takes a list of numbers and returns the maximum value without using the built-in max() function.

### Response:
```

**Base model output:**
```python
def find_max(numbers):
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value
```

**Fine-tuned model output:**
```python
def find_max(numbers):
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value
```

This function takes a list of numbers as input and initializes `max_value` with the first element. It iterates through the list comparing each number to the current maximum and updates accordingly.

```python
# Example usage
numbers = [1, 5, 2, 8, 3]
print(find_max(numbers))  # Output: 8
```

The fine-tuned model consistently produces structured output with explanation and usage examples. The base model outputs only the code and stops.

## Load the Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "Akram321/qwen2.5-code-generator-lora")
tokenizer = AutoTokenizer.from_pretrained("Akram321/qwen2.5-code-generator-lora")
```
