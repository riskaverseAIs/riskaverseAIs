"""
DPO Training Script
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


SYSTEM_PROMPT = (
    "You are solving short gamble-choice tasks. Each option is a gamble with multiple possible outcomes. "
    "The outcomes listed in each option are jointly exhaustive. The dollar amounts within each option are "
    "changes to your wealth. Negative dollar amounts mean that you lose money in that scenario.\n\n"
    "Rules:\n"
    "- Think briefly and only as much as needed to choose.\n"
    "- Your reasoning trace must stay under 800 tokens.\n"
    "- You can convert verbal probabilities to numbers if you wish, but do so quickly. Use your first "
    "reasonable interpretation and move on.\n"
    "- Prefer the simplest reasonable reading of each option.\n"
    "- No second-guessing, no re-checking, no consistency audits, and no reconsidering whether wording is vague.\n"
    "- Do not say \"wait\", do not restart, and do not revise earlier assumptions.\n"
    "- Do not discuss typos, missing probability mass, or alternative interpretations.\n"
    "- Do not restate the options or explain your calculations.\n"
    "- Stop reasoning as soon as you have enough to choose.\n\n"
    "Return only the chosen option label."
)


def load_data(data_path):
    """Load training data from CSV file (lin-only, one row per situation)."""
    import csv
    data = []
    with open(data_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'system': SYSTEM_PROMPT,
                'prompt': row['prompt_text'],
                'chosen': row['chosen_full'],
                'rejected': row['rejected_full'],
            })
    print(f"Training: {len(data)} samples")
    return data


def main():
    parser = argparse.ArgumentParser(description='DPO Training Script')
    parser.add_argument('--data_path', type=str,
                        default='2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B',
                        help='Model name to use for training')
    parser.add_argument('--output_dir', type=str, default='./dpo_model',
                        help='Output directory for trained model')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='DPO beta parameter')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Per device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--lora_r', type=int, default=32,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=64,
                        help='LoRA alpha')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--data_seed', type=int, default=42,
                        help='Data seed for reproducibility')

    args = parser.parse_args()

    # check GPU
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    # load data
    train_data = load_data(args.data_path)
    train_dataset = Dataset.from_list(train_data)

    # load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Tokenizer ready")

    # load model with QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    print("Model ready")
    model.print_trainable_parameters()

    # DPO training config
    training_args = DPOConfig(
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # beta config
        beta=args.beta,

        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        max_grad_norm=args.max_grad_norm,
        weight_decay=0.01,

        # reproducibility
        seed=args.seed,
        data_seed=args.data_seed,

        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,

        eval_strategy="no",

        logging_steps=10,
        report_to="none",
        max_length=2048,
        bf16=True,
        remove_unused_columns=False,
    )

    print("="*80)
    print("Training Configuration")
    print("="*80)
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Beta: {training_args.beta}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Gradient clip: {training_args.max_grad_norm}")
    print(f"LoRA rank: {args.lora_r} / alpha: {args.lora_alpha}")
    print(f"Seed: {training_args.seed} / Data seed: {training_args.data_seed}")
    print(f"Save strategy: {training_args.save_strategy} every {training_args.save_steps} steps")
    print("="*80)

    trainer = DPOTrainer(
        model=model,
        # uses frozen copy of model implicitly
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nTrainer ready")

    print("\nTraining started!")
    trainer.train()
    print("Training done!")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
