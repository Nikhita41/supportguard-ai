Create a concise, clean README.md for the project.

# SupportGuard AI

An AI-powered customer support system using a LoRA fine-tuned Qwen2.5-0.5B-Instruct model for intent classification, combined with a TF-IDF based guardrail and grounded response generation.

## Features
- 27-class customer intent classification
- LoRA fine-tuning with PEFT
- TF-IDF + cosine similarity guardrail
- Grounded response generation
- End-to-end customer support pipeline

## Dataset
Bitext Customer Support Training Dataset (27K).

## Results
- 97.5% accuracy on 1,000 held-out test samples
- 34/34 guardrail tests passed

## Tech Stack
Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, scikit-learn, Qwen2.5-0.5B-Instruct

## Run

```bash
python src/inference.py
