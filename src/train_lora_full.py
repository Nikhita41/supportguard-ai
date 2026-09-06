"""
train_lora_full.py  —  MVP CPU run
-----------------------------------
Fine-tunes Qwen/Qwen2.5-0.5B-Instruct with PEFT LoRA on a stratified
5,000-example sample drawn from the full Bitext customer-support dataset.

Data pipeline:
  - Load full dataset, shuffle (random_state=42)
  - 80 / 10 / 10  train / validation / test split (stratified)
  - Stratified-sample 5,000 rows from the 80% train pool for this MVP run
  - Validation and test splits are kept completely separate
  - Evaluate once at the end of the epoch

Checkpointing:
  - Saves a checkpoint every 250 steps → safe to interrupt and resume
  - Re-run the same command to resume from the latest checkpoint automatically

Saves the final adapter to  models/lora_mvp  (lora_test is untouched).

Run from the project root:
    python src/train_lora_full.py
"""

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
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
ROOT       = Path(__file__).parent.parent
DATA_CSV   = ROOT / "data" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
OUTPUT_DIR = ROOT / "models" / "lora_mvp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
NUM_TRAIN_SAMPLES = 5_000   # MVP: stratified sample from the 80% train pool
NUM_VAL_SAMPLES   = 200     # MVP: stratified sample from the 10% validation pool

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ---------------------------------------------------------------------------
# 1. Load, shuffle, and split dataset  (80 / 10 / 10)
# ---------------------------------------------------------------------------
print("Loading dataset...")
raw_df = pd.read_csv(DATA_CSV)

df = raw_df[["instruction", "intent"]].copy()

# Shuffle the full dataset before splitting
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# First split: 80% train pool | 20% temp
train_pool_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["intent"],
)

# Second split: temp → 50% val pool | 50% test  (10% + 10% of total)
val_pool_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["intent"],
)

# MVP: Stratified 5,000-row sample from the 80% training pool (~21,497 rows)
train_df, _ = train_test_split(
    train_pool_df,
    train_size=NUM_TRAIN_SAMPLES,
    random_state=RANDOM_STATE,
    stratify=train_pool_df["intent"],
)
train_df = train_df.reset_index(drop=True)

# MVP: Stratified 200-row sample from the validation pool (~2,687 rows)
val_df, _ = train_test_split(
    val_pool_df,
    train_size=NUM_VAL_SAMPLES,
    random_state=RANDOM_STATE,
    stratify=val_pool_df["intent"],
)
val_df = val_df.reset_index(drop=True)

# Test set remains completely untouched
test_df = test_df.reset_index(drop=True)

print(f"\nDataset sizes:")
print(f"  Training   : {len(train_df):,}")
print(f"  Validation : {len(val_df):,}")
print(f"  Test       : {len(test_df):,} held out")
print()


def format_text(sub_df: pd.DataFrame) -> Dataset:
    """Apply the training prompt template and convert to HuggingFace Dataset."""
    out = sub_df.copy()
    out["text"] = (
        "Classify the customer request into the correct intent.\n\n"
        "Customer request: " + out["instruction"]
        + "\n\nIntent: " + out["intent"]
    )
    return Dataset.from_pandas(out[["text"]].reset_index(drop=True))


train_dataset = format_text(train_df)
val_dataset   = format_text(val_df)
# test_df is held out — not used during training

# ---------------------------------------------------------------------------
# 2. Tokenizer
# ---------------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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


print("Tokenizing train split...")
tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

print("Tokenizing validation split...")
tokenized_val = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

# ---------------------------------------------------------------------------
# 3. Model + LoRA adapters
# ---------------------------------------------------------------------------
print("\nLoading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    trust_remote_code=True,
)
model.config.use_cache = False

print("Attaching LoRA adapters...")
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# 4. Training arguments
# ---------------------------------------------------------------------------
# warmup ~5% of total optimizer steps
total_steps  = (len(tokenized_train) // (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
warmup_steps = max(1, int(total_steps * 0.05))

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    # Evaluation: once at end of epoch (not every N steps)
    eval_strategy="epoch",
    # Checkpointing: save every 250 steps so interrupted runs can be resumed
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,           # keep the 3 most recent checkpoints
    load_best_model_at_end=False, # keeps RAM usage low on CPU
    # Logging
    logging_steps=50,
    report_to="none",
    # CPU-only flags
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    use_cpu=True,
)

# ---------------------------------------------------------------------------
# 5. Data collator
# ---------------------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

# ---------------------------------------------------------------------------
# 6. Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ---------------------------------------------------------------------------
# 7. Train  (resumes from latest checkpoint automatically if one exists)
# ---------------------------------------------------------------------------
# Check if a checkpoint already exists so an interrupted run continues cleanly
checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
resume_path = str(checkpoints[-1]) if checkpoints else None

if resume_path:
    print(f"\nResuming from checkpoint: {resume_path}\n")
else:
    print(f"\nStarting fresh MVP LoRA run ({len(tokenized_train)} train examples, 1 epoch)...")

print(f"Validation will run once at the end of the epoch ({len(tokenized_val)} examples).")
print("Checkpoints saved every 250 steps — safe to interrupt and re-run.\n")

trainer.train(resume_from_checkpoint=resume_path)

# ---------------------------------------------------------------------------
# 8. Save final LoRA adapter
# ---------------------------------------------------------------------------
print(f"\nSaving LoRA adapter to: {OUTPUT_DIR}")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\nDone!")
print(f"  models/lora_test  — original 500-sample adapter  (unchanged)")
print(f"  models/lora_mvp   — this 5,000-sample MVP adapter")
print("\nTo resume later if interrupted, just re-run:")
print("    python src/train_lora_full.py")
print("\nTo load the adapter:")
print(f"  model = PeftModel.from_pretrained(base_model, r'{OUTPUT_DIR}')")
