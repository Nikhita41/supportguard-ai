"""
train_lora_continue.py
----------------------
Continues training the existing LoRA adapter from models/lora_mvp on a NEW,
non-overlapping 5,000-example batch from the 80% Bitext training pool.

Key guarantees:
  1. Loads models/lora_mvp (does NOT restart from base model weights).
  2. Samples 5,000 NEW stratified examples from remaining training pool (0 overlap).
  3. Uses the exact same 200 validation samples for evaluation (random_state=42).
  4. Keeps the 2,688 test set completely untouched and held-out.
  5. Checkpoints every 250 steps for interrupt/resume safety.
  6. Saves resulting adapter to models/lora_10k (models/lora_mvp and lora_test untouched).
  7. Computes and prints Accuracy, Precision, Recall, F1, and Confusion Matrix on the 200 val examples.

Run from project root:
    python src/train_lora_continue.py
"""

from pathlib import Path
import re
import pandas as pd
import torch
from datasets import Dataset
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT               = Path(__file__).parent.parent
DATA_CSV           = ROOT / "data" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
INPUT_ADAPTER_DIR  = ROOT / "models" / "lora_mvp"
OUTPUT_ADAPTER_DIR = ROOT / "models" / "lora_10k"
OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
MODEL_NAME        = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH        = 128
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 1
BATCH_SIZE        = 1
GRAD_ACCUM_STEPS  = 4
RANDOM_STATE      = 42
NUM_TRAIN_SAMPLES = 5_000
NUM_VAL_SAMPLES   = 200

# ---------------------------------------------------------------------------
# 1. Dataset Partitioning & Non-Overlapping Sampling
# ---------------------------------------------------------------------------
print("=" * 70)
print(" Continued LoRA Training: Stage 2 (5k -> 10k total examples)")
print("=" * 70)

if not INPUT_ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"Base adapter '{INPUT_ADAPTER_DIR}' not found. "
        "Please make sure models/lora_mvp exists before continuing training."
    )

print(f"\n[1/6] Loading and partitioning dataset from: {DATA_CSV.name}")
raw_df = pd.read_csv(DATA_CSV)
df = raw_df[["instruction", "intent"]].copy()

# Shuffle full dataset identically (random_state=42)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 80% Train pool | 20% Temp
train_pool_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["intent"],
)

# Temp -> 50% Val pool | 50% Test pool (10% + 10% of total)
val_pool_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["intent"],
)

# Step A: Recreate the EXACT 5,000 samples used in Stage 1 (MVP)
stage1_train_df, remaining_train_df = train_test_split(
    train_pool_df,
    train_size=NUM_TRAIN_SAMPLES,
    random_state=RANDOM_STATE,
    stratify=train_pool_df["intent"],
)

# Step B: Sample 5,000 NEW stratified examples from remaining training pool
stage2_train_df, remaining_train_pool_after = train_test_split(
    remaining_train_df,
    train_size=NUM_TRAIN_SAMPLES,
    random_state=RANDOM_STATE,
    stratify=remaining_train_df["intent"],
)

# Step C: Recreate the EXACT identical 200 validation samples
val_df, _ = train_test_split(
    val_pool_df,
    train_size=NUM_VAL_SAMPLES,
    random_state=RANDOM_STATE,
    stratify=val_pool_df["intent"],
)

# Clean indices
stage1_train_df = stage1_train_df.reset_index(drop=False)
stage2_train_df = stage2_train_df.reset_index(drop=False)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Strict Verification: Check Overlap between Stage 1 and Stage 2
# ---------------------------------------------------------------------------
stage1_indices = set(stage1_train_df["index"])
stage2_indices = set(stage2_train_df["index"])
overlap_count = len(stage1_indices.intersection(stage2_indices))

print("\n--- DATASET SPLIT & OVERLAP VERIFICATION ---")
print(f"  Total Dataset Rows         : {len(df):,}")
print(f"  Full 80% Train Pool        : {len(train_pool_df):,} rows")
print(f"  Stage 1 Samples (used in MVP): {len(stage1_train_df):,} rows")
print(f"  Stage 2 Samples (NEW)      : {len(stage2_train_df):,} rows")
print(f"  Remaining Train Pool       : {len(remaining_train_pool_after):,} rows")
print(f"  Validation Samples (fixed) : {len(val_df):,} rows")
print(f"  Test Set (HELD OUT)        : {len(test_df):,} rows")
print(f"  Overlap (Stage 1 vs Stage 2): {overlap_count} records (100% DISJOINT)")
print("-------------------------------------------\n")

assert overlap_count == 0, "Error: Overlap detected between training stages!"


def format_text(sub_df: pd.DataFrame) -> Dataset:
    """Apply the training prompt template and convert to HuggingFace Dataset."""
    out = sub_df.copy()
    out["text"] = (
        "Classify the customer request into the correct intent.\n\n"
        "Customer request: " + out["instruction"]
        + "\n\nIntent: " + out["intent"]
    )
    return Dataset.from_pandas(out[["text"]].reset_index(drop=True))


train_dataset = format_text(stage2_train_df)
val_dataset   = format_text(val_df)

# ---------------------------------------------------------------------------
# 2. Tokenizer
# ---------------------------------------------------------------------------
print("[2/6] Loading tokenizer from models/lora_mvp...")
tokenizer = AutoTokenizer.from_pretrained(str(INPUT_ADAPTER_DIR), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


print("  Tokenizing Stage 2 train split...")
tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

print("  Tokenizing validation split...")
tokenized_val = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

# ---------------------------------------------------------------------------
# 3. Load Base Model & Attach Existing LoRA Adapter in Trainable Mode
# ---------------------------------------------------------------------------
print(f"\n[3/6] Loading base model: {MODEL_NAME}...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    trust_remote_code=True,
)
base_model.config.use_cache = False

print(f"  Loading existing LoRA adapter from: {INPUT_ADAPTER_DIR} (is_trainable=True)...")
model = PeftModel.from_pretrained(
    base_model,
    str(INPUT_ADAPTER_DIR),
    is_trainable=True,
)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# 4. Training Arguments & Checkpointing
# ---------------------------------------------------------------------------
total_steps  = (len(tokenized_train) // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
warmup_steps = max(1, int(total_steps * 0.05))

training_args = TrainingArguments(
    output_dir=str(OUTPUT_ADAPTER_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,
    load_best_model_at_end=False,
    logging_steps=50,
    report_to="none",
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    use_cpu=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ---------------------------------------------------------------------------
# 5. Train / Resume
# ---------------------------------------------------------------------------
checkpoints = sorted(OUTPUT_ADAPTER_DIR.glob("checkpoint-*"))
resume_path = str(checkpoints[-1]) if checkpoints else None

if resume_path:
    print(f"\n[4/6] Resuming Stage 2 training from checkpoint: {resume_path}")
else:
    print(f"\n[4/6] Starting Stage 2 fine-tuning on 5,000 NEW examples...")

print("  Checkpoints will be saved to: models/lora_10k")
trainer.train(resume_from_checkpoint=resume_path)

# ---------------------------------------------------------------------------
# 6. Save Continued Adapter
# ---------------------------------------------------------------------------
print(f"\n[5/6] Saving continued LoRA adapter to: {OUTPUT_ADAPTER_DIR}")
model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))

# ---------------------------------------------------------------------------
# 7. Post-Training Evaluation on 200 Validation Examples
# ---------------------------------------------------------------------------
print("\n[6/6] Running Post-Training Evaluation on the 200 Validation Examples...")
model.eval()

newline_token_id = tokenizer.encode("\n", add_special_tokens=False)
stop_token_ids = [tokenizer.eos_token_id] + newline_token_id

instructions = val_df["instruction"].tolist()
true_intents = val_df["intent"].tolist()
predicted_intents = []

def build_prompt(user_input: str) -> str:
    return (
        "Classify the customer request into the correct intent.\n\n"
        f"Customer request: {user_input}\n\n"
        "Intent:"
    )

def extract_intent(raw_text: str) -> str:
    clean = raw_text.strip()
    match = re.match(r"[a-zA-Z][a-zA-Z_]*", clean)
    return match.group(0) if match else clean

# Evaluation loop
for user_input in instructions:
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True)
    predicted_intents.append(extract_intent(raw_output))

# Metrics
acc = accuracy_score(true_intents, predicted_intents)
prec_weighted = precision_score(true_intents, predicted_intents, average="weighted", zero_division=0)
rec_weighted = recall_score(true_intents, predicted_intents, average="weighted", zero_division=0)
f1_weighted = f1_score(true_intents, predicted_intents, average="weighted", zero_division=0)

prec_macro = precision_score(true_intents, predicted_intents, average="macro", zero_division=0)
rec_macro = recall_score(true_intents, predicted_intents, average="macro", zero_division=0)
f1_macro = f1_score(true_intents, predicted_intents, average="macro", zero_division=0)

print("\n" + "=" * 70)
print(" STAGE 2 (10k TOTAL) VALIDATION RESULTS (200 EXAMPLES)")
print("=" * 70)
print(f"  Accuracy (Overall)   : {acc * 100:.2f}%")
print(f"  Precision (Weighted) : {prec_weighted * 100:.2f}%  |  Macro: {prec_macro * 100:.2f}%")
print(f"  Recall (Weighted)    : {rec_weighted * 100:.2f}%  |  Macro: {rec_macro * 100:.2f}%")
print(f"  F1-Score (Weighted)  : {f1_weighted * 100:.2f}%  |  Macro: {f1_macro * 100:.2f}%")
print("=" * 70)

print("\n" + "=" * 70)
print(" DETAILED CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(true_intents, predicted_intents, zero_division=0))

# Confusion Matrix
unique_labels = sorted(list(set(true_intents + predicted_intents)))
cm = confusion_matrix(true_intents, predicted_intents, labels=unique_labels)
print("=" * 70)
print(" CONFUSION MATRIX BREAKDOWN (Per Class)")
print("=" * 70)
correct_per_class = cm.diagonal()
total_per_class = cm.sum(axis=1)
for label, correct, total in zip(unique_labels, correct_per_class, total_per_class):
    if total > 0:
        pct = (correct / total) * 100
        print(f"  {label:<30} : {correct:>2}/{total:<2} correct ({pct:.1f}%)")

print("\n" + "=" * 70)
print(" TRAINING COMPLETE")
print("=" * 70)
print("  Original MVP Adapter : models/lora_mvp  (5,000 examples, UNCHANGED)")
print("  New 10k Adapter      : models/lora_10k  (10,000 total examples)")
print("  Test Set             : 2,688 records    (HELD OUT & UNTOUCHED)")
print("=" * 70)
