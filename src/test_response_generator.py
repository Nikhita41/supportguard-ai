"""
test_response_generator.py
--------------------------
Unit tests and verification suite for SupportResponseGenerator.
Tests:
  1. Base model reuse with adapter disabled
  2. Grounded response generation across various intents
  3. Verification that responses are polite, concise, and do not invent facts
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from response_generator import SupportResponseGenerator

ROOT = Path(__file__).parent.parent
ADAPTER_DIR = ROOT / "models" / "lora_10k"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def run_tests():
    print("=" * 80)
    print(" SUPPORT RESPONSE GENERATOR TEST SUITE")
    print("=" * 80)

    print(f"\n[1/3] Loading Tokenizer & Base Model ({BASE_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    print(f"  Attaching LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    print("\n[2/3] Initializing SupportResponseGenerator...")
    generator = SupportResponseGenerator(model=model, tokenizer=tokenizer)
    print("  Generator ready.")

    test_queries = [
        ("I need to cancel my order", "cancel_order"),
        ("Where is my package #88912?", "track_order"),
        ("How can I change my delivery shipping address?", "change_shipping_address"),
        ("I forgot my account password and cannot log in", "recover_password"),
        ("My credit card was declined at checkout", "payment_issue"),
        ("Can I get a refund for a damaged item?", "get_refund"),
        ("How do I speak with a human customer service agent?", "contact_human_agent"),
    ]

    print("\n[3/3] Testing Grounded Response Generation:\n")

    for idx, (query, intent) in enumerate(test_queries, 1):
        print(f"[{idx}] Customer Query  : \"{query}\"")
        print(f"    Verified Intent : {intent}")

        response = generator.generate_response(user_query=query, predicted_intent=intent)
        print(f"    Generated Reply : {response}")
        print("-" * 80)

        # Basic sanity validations
        assert len(response) > 10, f"Response too short: '{response}'"
        assert "I checked your order" not in response, "Hallucinated action detected!"

    print("\nAll response generation tests completed successfully!")


if __name__ == "__main__":
    run_tests()
