"""
final_test_evaluation.py
------------------------
Performs the final benchmark evaluation of the models/lora_10k adapter on a
stratified 1,000-example sample from the untouched 2,688 held-out test set.

Guarantees:
  1. Zero overlap with the 10,000 training examples (Stage 1 + Stage 2).
  2. Zero overlap with the 200 validation samples.
  3. Uses target-intent prefix matching against all 27 classes.
  4. Computes Accuracy, Weighted & Macro Precision/Recall/F1, Classification Report,
     Confusion Matrix, and Misclassifications.
  5. Pure evaluation mode (torch.no_grad()) — no model modification or retraining.

Run from project root:
    python src/final_test_evaluation.py
"""

from pathlib import Path
import re
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
DATA_CSV    = ROOT / "data" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
ADAPTER_DIR = ROOT / "models" / "lora_10k"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
RANDOM_STATE     = 42
NUM_TEST_SAMPLES = 1_000
BATCH_SIZE       = 16      # Batched inference for CPU efficiency
MAX_LENGTH       = 128
MAX_NEW_TOKENS   = 10

VALID_INTENTS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice",
    "get_refund", "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review", "set_up_shipping_address",
    "switch_account", "track_order", "track_refund"
]
SORTED_VALID_INTENTS = sorted(VALID_INTENTS, key=len, reverse=True)


def build_prompt(user_input: str) -> str:
    return (
        "Classify the customer request into the correct intent.\n\n"
        f"Customer request: {user_input}\n\n"
        "Intent:"
    )


def extract_intent(raw_text: str) -> str:
    clean = raw_text.strip()
    for intent in SORTED_VALID_INTENTS:
        if clean.startswith(intent):
            return intent
    match = re.match(r"[a-zA-Z][a-zA-Z_]*", clean)
    return match.group(0) if match else clean


def main():
    print("=" * 75)
    print(" FINAL BENCHMARK EVALUATION: 10K LoRA Model (1,000 Test Examples)")
    print("=" * 75)

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter '{ADAPTER_DIR}' not found.")

    # -----------------------------------------------------------------------
    # 1. Recreate Identical Splits and Verify Zero Overlap
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Loading and partitioning dataset from: {DATA_CSV.name}")
    raw_df = pd.read_csv(DATA_CSV)
    df = raw_df[["instruction", "intent"]].copy()

    # Shuffle full dataset
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 80% Train Pool | 20% Temp
    train_pool_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["intent"],
    )

    # Temp -> 50% Val Pool | 50% Test Pool (10% + 10% of total)
    val_pool_df, test_pool_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["intent"],
    )

    # Recreate Stage 1 (5k) and Stage 2 (5k) training samples
    stage1_train_df, rem_train_df = train_test_split(
        train_pool_df,
        train_size=5_000,
        random_state=RANDOM_STATE,
        stratify=train_pool_df["intent"],
    )
    stage2_train_df, _ = train_test_split(
        rem_train_df,
        train_size=5_000,
        random_state=RANDOM_STATE,
        stratify=rem_train_df["intent"],
    )

    # Recreate the 200 validation samples
    val_200_df, _ = train_test_split(
        val_pool_df,
        train_size=200,
        random_state=RANDOM_STATE,
        stratify=val_pool_df["intent"],
    )

    # Sample 1,000 stratified examples from the 2,688 test pool
    test_1000_df, _ = train_test_split(
        test_pool_df,
        train_size=NUM_TEST_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=test_pool_df["intent"],
    )

    # Preserve original indices for overlap assertions
    train_10k_indices = set(stage1_train_df.index).union(set(stage2_train_df.index))
    val_200_indices   = set(val_200_df.index)
    test_1000_indices = set(test_1000_df.index)

    overlap_with_train = len(test_1000_indices.intersection(train_10k_indices))
    overlap_with_val   = len(test_1000_indices.intersection(val_200_indices))

    print("\n--- DATASET PARTITION & OVERLAP VERIFICATION ---")
    print(f"  Total Dataset Records           : {len(df):,}")
    print(f"  10K Training Set (Stage 1 & 2)  : {len(train_10k_indices):,} rows")
    print(f"  Validation Set                  : {len(val_200_indices):,} rows")
    print(f"  Total Held-Out Test Pool        : {len(test_pool_df):,} rows")
    print(f"  Sampled Final Test Evaluation   : {len(test_1000_df):,} rows (stratified across 27 intents)")
    print(f"  Overlap with Training Data      : {overlap_with_train} rows (100% DISJOINT)")
    print(f"  Overlap with Validation Data    : {overlap_with_val} rows (100% DISJOINT)")
    print("------------------------------------------------\n")

    assert overlap_with_train == 0, "FATAL: Overlap detected between test and training data!"
    assert overlap_with_val == 0, "FATAL: Overlap detected between test and validation data!"

    test_1000_df = test_1000_df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 2. Load Model & LoRA 10k Adapter
    # -----------------------------------------------------------------------
    print(f"[2/5] Loading tokenizer & base model ({BASE_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    print(f"  Attaching LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)
    stop_token_ids = [tokenizer.eos_token_id] + newline_token_id

    # -----------------------------------------------------------------------
    # 3. Batched Inference on 1,000 Test Examples
    # -----------------------------------------------------------------------
    print(f"\n[3/5] Running batched inference on {len(test_1000_df):,} test examples (Batch Size: {BATCH_SIZE})...")

    instructions = test_1000_df["instruction"].tolist()
    true_intents = test_1000_df["intent"].tolist()
    predicted_intents = []

    for i in tqdm(range(0, len(instructions), BATCH_SIZE), desc="Evaluating Test Set"):
        batch_instructions = instructions[i : i + BATCH_SIZE]
        batch_prompts = [build_prompt(text) for text in batch_instructions]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[:, prompt_len:]
        raw_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for raw_text in raw_outputs:
            predicted_intents.append(extract_intent(raw_text))

    # -----------------------------------------------------------------------
    # 4. Compute Metrics
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Computing final evaluation metrics...")

    acc = accuracy_score(true_intents, predicted_intents) * 100
    prec_w = precision_score(true_intents, predicted_intents, average="weighted", zero_division=0) * 100
    rec_w = recall_score(true_intents, predicted_intents, average="weighted", zero_division=0) * 100
    f1_w = f1_score(true_intents, predicted_intents, average="weighted", zero_division=0) * 100

    prec_m = precision_score(true_intents, predicted_intents, average="macro", zero_division=0) * 100
    rec_m = recall_score(true_intents, predicted_intents, average="macro", zero_division=0) * 100
    f1_m = f1_score(true_intents, predicted_intents, average="macro", zero_division=0) * 100

    print("\n" + "=" * 75)
    print(" FINAL TEST BENCHMARK RESULTS (1,000 UNSEEN TEST SAMPLES)")
    print("=" * 75)
    print(f"  Accuracy (Overall)   : {acc:.2f}%")
    print(f"  Weighted Precision   : {prec_w:.2f}%")
    print(f"  Weighted Recall      : {rec_w:.2f}%")
    print(f"  Weighted F1-Score    : {f1_w:.2f}%")
    print("-" * 75)
    print(f"  Macro Precision      : {prec_m:.2f}%")
    print(f"  Macro Recall         : {rec_m:.2f}%")
    print(f"  Macro F1-Score       : {f1_m:.2f}%")
    print("=" * 75)

    print("\n" + "=" * 75)
    print(" CLASSIFICATION REPORT (Per-Intent Precision, Recall, F1)")
    print("=" * 75)
    print(classification_report(true_intents, predicted_intents, zero_division=0))

    # Confusion Matrix
    unique_labels = sorted(list(set(true_intents + predicted_intents)))
    cm = confusion_matrix(true_intents, predicted_intents, labels=unique_labels)

    print("=" * 75)
    print(" CONFUSION MATRIX BREAKDOWN (Per Class Correctness)")
    print("=" * 75)
    correct_per_class = cm.diagonal()
    total_per_class = cm.sum(axis=1)
    for label, correct, total in zip(unique_labels, correct_per_class, total_per_class):
        if total > 0:
            pct = (correct / total) * 100
            print(f"  {label:<30} : {correct:>3}/{total:<3} correct ({pct:6.2f}%)")

    # -----------------------------------------------------------------------
    # 5. Misclassification Analysis
    # -----------------------------------------------------------------------
    errors = []
    for inst, true_i, pred_i in zip(instructions, true_intents, predicted_intents):
        if true_i != pred_i:
            errors.append((inst, true_i, pred_i))

    print("\n" + "=" * 75)
    print(" MISCLASSIFICATION ANALYSIS")
    print("=" * 75)
    if not errors:
        print("  None! Model achieved 100% accuracy on all 1,000 test examples.")
    else:
        print(f"  Total Misclassifications: {len(errors)} / {len(test_1000_df)} ({len(errors)/len(test_1000_df)*100:.2f}% error rate)\n")
        for idx, (inst, true_i, pred_i) in enumerate(errors, 1):
            print(f"  [{idx}] Instruction : {inst}")
            print(f"      True Intent : {true_i}")
            print(f"      Pred Intent : {pred_i}\n")

    print("=" * 75)
    print(" Final evaluation completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
