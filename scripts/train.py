import argparse
import json
import os
import time
import yaml

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def create_dummy_files_if_needed(config_path, train_path, val_path):
    """Creates dummy config and data files if they don't exist."""
    # Create dummy config file
    if not os.path.exists(config_path):
        print(f"'{config_path}' not found. Creating a default one.")
        default_config = {
            'model_name': 'Qwen/Qwen2-4B-Instruct',
            'train_data_path': train_path,
            'val_data_path': val_path,
            'output_dir': 'qwen2-4b-finetuned-adapter',
            'max_seq_length': 2048,
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)

    # Create dummy data files
    os.makedirs("data", exist_ok=True)
    dummy_data_sample = {
        "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>"
    }
    if not os.path.exists(train_path):
        print(f"'{train_path}' not found. Creating a dummy training file.")
        with open(train_path, 'w') as f:
            for _ in range(100):
                f.write(json.dumps(dummy_data_sample) + '\n')
    if not os.path.exists(val_path):
        print(f"'{val_path}' not found. Creating a dummy validation file.")
        with open(val_path, 'w') as f:
            for _ in range(100):
                f.write(json.dumps(dummy_data_sample) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B using QLoRA (PatentFlow-verified pipeline).")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode for a quick check.")
    args = parser.parse_args()

    # Create dummy files for a quick start if they don't exist
    create_dummy_files_if_needed(args.config, "data/train.jsonl", "data/val.jsonl")

    # Load configuration from YAML
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # --- 1. Model and Tokenizer Loading ---
    print("--- 1. Loading model and tokenizer ---")
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True,
        padding_side='right',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. LoRA Configuration ---
    print("--- 2. Setting up LoRA configuration ---")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # --- 3. Data Loading and Preparation ---
    print("--- 3. Loading and preparing datasets ---")
    train_dataset = load_dataset('json', data_files=config['train_data_path'], split='train')
    val_dataset = load_dataset('json', data_files=config['val_data_path'], split='train')

    if args.smoke:
        print("--- Running in smoke test mode ---")
        train_dataset = train_dataset.select(range(50))
        val_dataset = val_dataset.select(range(50))

    # --- 4. Trainer Configuration ---
    print("--- 4. Configuring the SFTTrainer ---")
    
    # Data collator for completion-only fine-tuning
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Training arguments
    if args.smoke:
        training_args = SFTConfig(
            output_dir=config['output_dir'],
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            logging_steps=10,
            learning_rate=1e-4,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            disable_tqdm=False,
            gradient_checkpointing=True,
            save_strategy="no",
            eval_strategy="no",
        )
    else:
        training_args = SFTConfig(
            output_dir=config['output_dir'],
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            logging_steps=25,
            learning_rate=1e-4,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
        )

    # --- 5. Training ---
    print("--- 5. Initializing and starting training ---")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=config['max_seq_length'],
        packing=False,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"--- Training finished in {end_time - start_time:.2f} seconds ---")

    # --- 6. Save Model and VRAM Usage ---
    print("--- 6. Saving final adapter and reporting VRAM usage ---")
    trainer.save_model(config['output_dir'])
    
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\nPeak VRAM usage during training: {peak_vram_gb:.2f} GB")

    # --- 7. Inference Test ---
    print("\n--- 7. Running a quick generation test ---")
    del model
    del trainer
    torch.cuda.empty_cache()

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load PEFT model with trained adapter
    peft_model = PeftModel.from_pretrained(base_model, config['output_dir'])
    
    # Merge adapter and unload for faster inference
    merged_model = peft_model.merge_and_unload()

    # Prepare prompt
    test_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a short story about a robot who discovers music.<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(test_prompt, return_tensors="pt", return_attention_mask=True).to(merged_model.device)

    print("\nPrompt:")
    print(test_prompt)
    print("\nGenerating response...")

    # Generate response
    outputs = merged_model.generate(
        **inputs,
        max_new_tokens=150,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("\nFull Model Output:")
    print(response)

    # Extract only the assistant's response
    assistant_response = response.split(response_template)[-1]
    print("\nAssistant's Response:")
    print(assistant_response.replace("<|im_end|>", "").strip())


if __name__ == "__main__":
    main()