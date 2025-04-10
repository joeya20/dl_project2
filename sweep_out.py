import os
import pandas as pd
import torch
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset, ClassLabel
import pickle
from sklearn.metrics import accuracy_score
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc # Garbage collector for potentially clearing GPU memory


base_model = "roberta-base"
output_base_dir = "dse_results" # Base directory for all DSE runs
os.makedirs(output_base_dir, exist_ok=True)

# --- Hyperparameter Ranges for DSE ---
learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
lora_ranks = [4, 5, 6, 7] 
lora_alpha_scaling = [2, 3, 4]

# --- Data Loading and Preprocessing (Done Once) ---
print("Loading and preprocessing data...")
dataset = load_dataset("ag_news", split="train")
tokenizer = RobertaTokenizer.from_pretrained(base_model)

def preprocess(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=512) # Added max_length
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Extract the number of classess and their names
num_labels = dataset.features["label"].num_classes
class_names = dataset.features["label"].names
print(f"Number of labels: {num_labels}")
print(f"The labels: {class_names}")

# Create an id2label mapping
id2label = {i: label for i, label in enumerate(class_names)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Split the original training set (Done Once)
split_datasets = tokenized_dataset.train_test_split(test_size=0.1, seed=42) # Using a percentage split
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- Helper Functions ---

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

def evaluate_model(
    inference_model, dataset, labelled=True, batch_size=64, data_collator=None
):
    """
    Evaluate a PEFT model on a dataset. Simplified version for DSE.
    Returns evaluation metric (accuracy) if labelled, else predictions.
    """
    eval_dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    all_labels = [] # Store labels only if needed for metrics

    if labelled:
        metric = evaluate.load("accuracy") # Use evaluate library for consistency

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # Ensure batch is on the correct device
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = inference_model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())

        if labelled:
            # Check if 'labels' key exists and move to CPU
            if "labels" in batch:
                 references = batch["labels"].cpu()
                 all_labels.append(references)
                 metric.add_batch(predictions=predictions.cpu(), references=references)
            else:
                print("Warning: 'labels' key not found in evaluation batch.")

    all_predictions = torch.cat(all_predictions, dim=0)

    if labelled:
        try:
            eval_metric = metric.compute()
            print(f"Evaluation Accuracy: {eval_metric['accuracy']:.4f}")
            return eval_metric, all_predictions
        except Exception as e:
             print(f"Error computing metric: {e}")
             # Optionally return NaN or raise error if labels were expected but missing/empty
             if not all_labels: # Check if any labels were collected
                  print("No labels collected during evaluation.")
             return {"accuracy": float('nan')}, all_predictions # Return NaN if metric fails
    else:
        return all_predictions


# --- Design Space Exploration Loop ---
results = []

for lr in learning_rates:
    for rank in lora_ranks:
        for alpha_scale in lora_alpha_scaling:
            alpha = rank * alpha_scale
            run_name = f"lr_{lr}_rank_{rank}_alpha_{alpha}"
            print(f"\n--- Starting Run: {run_name} ---")

            # Define output directory for this specific run
            current_output_dir = os.path.join(output_base_dir, run_name)
            os.makedirs(current_output_dir, exist_ok=True)

            # 1. Load Base Model (Load fresh for each run)
            print("Loading base model...")
            model = RobertaForSequenceClassification.from_pretrained(
                base_model,
                id2label=id2label,
                num_labels=num_labels  # Ensure num_labels is passed correctly
            )

            # Move model to GPU before PEFT if possible
            if torch.cuda.is_available():
                model.to('cuda')

            # 2. Configure LoRA for this run
            print(f"Configuring LoRA with r={rank}, alpha={alpha}")
            peft_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.05,
                target_modules=["query", "key", "value"],
                bias="none",
                task_type="SEQ_CLS",
            )

            # 3. Apply PEFT to the model
            peft_model = get_peft_model(model, peft_config)
            print("PEFT Model Configured:")
            peft_model.print_trainable_parameters()

            # 4. Setup Training Arguments for this run
            training_args = TrainingArguments(
                output_dir=current_output_dir,
                report_to="none",
                eval_strategy="epoch", # Evaluate at the end of each epoch
                # eval_steps=200, # Or evaluate every N steps
                logging_steps=100,
                learning_rate=lr,
                num_train_epochs=1, # Keep epochs low for faster sweeps, adjust as needed
                # max_steps=500, # Or use max_steps for very quick tests
                per_device_train_batch_size=16, # Adjust based on GPU memory
                per_device_eval_batch_size=64, # Adjust based on GPU memory
                optim="adamw_torch",
                gradient_accumulation_steps=2, # Accumulate gradients to simulate larger batch size if needed
                fp16=torch.cuda.is_available(), # Enable mixed precision if using GPU
                # gradient_checkpointing=True, # Enable if memory is tight, slows down training
                # gradient_checkpointing_kwargs={"use_reentrant": False}, # Recommended setting for newer PyTorch
                save_strategy="epoch", # Save checkpoints less frequently during DSE
                save_total_limit=1, # Keep only the last checkpoint
                load_best_model_at_end=False, # Don't reload best model during sweep for speed
                dataloader_num_workers=4,
                seed=42, # Keep seed consistent for reproducibility across runs
                # Removed use_cpu=False, Trainer detects GPU automatically
            )

            # 5. Create Trainer for this run
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,  # Pass tokenizer for padding consistency if needed by trainer/collator
            )

            # 6. Train the model
            print("Starting training...")
            try:
                train_result = trainer.train()
                print("Training finished.")
                # trainer.save_model() # Optionally save the final model for the best run later

                # 7. Evaluate the model after training
                print("Evaluating model on evaluation set...")
                eval_metrics, _ = evaluate_model(
                    trainer.model,  # Use the trained model from the trainer
                    eval_dataset,
                    labelled=True,
                    batch_size=training_args.per_device_eval_batch_size,
                    data_collator=data_collator
                )
                final_accuracy = eval_metrics.get('accuracy', float('nan'))

            except Exception as e:
                print(f"!!! ERROR during training/evaluation for {run_name}: {e}")
                final_accuracy = float('nan')  # Record failure

            # 8. Store results
            results.append({
                "learning_rate": lr,
                "lora_rank": rank,
                "lora_alpha": alpha,
                "accuracy": final_accuracy,
                "output_dir": current_output_dir
            })
            print(f"Run {run_name} completed. Accuracy: {final_accuracy:.4f}")

            # 9. Clean up memory (Important!)
            del model
            del peft_model
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Post-DSE Analysis ---
print("\n--- Design Space Exploration Complete ---")

# Convert results to DataFrame for easy viewing/sorting
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="accuracy", ascending=False)

print("Top 5 Results:")
print(results_df.head())

# Save results to CSV
results_csv_path = os.path.join(output_base_dir, "dse_summary.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull DSE results saved to: {results_csv_path}")

# --- Optional: Run Inference with Best Model ---
if not results_df.empty and pd.notna(results_df.iloc[0]['accuracy']):
    best_run = results_df.iloc[0]
    print(f"\n--- Running inference with best model configuration ---")
    print(f"Best Config: LR={best_run['learning_rate']}, Rank={best_run['lora_rank']}, Alpha={best_run['lora_alpha']}")
    print(f"Best Accuracy: {best_run['accuracy']:.4f}")
    best_model_path = best_run['output_dir'] # Assumes best model saved here if save_strategy was used

    # Check if a checkpoint exists (Trainer saves checkpoints in subdirs)
    potential_checkpoint = os.path.join(best_model_path) # Trainer typically saves directly in output_dir if save_total_limit=1
    if os.path.exists(os.path.join(potential_checkpoint, "adapter_model.safetensors")) or \
       os.path.exists(os.path.join(potential_checkpoint, "adapter_model.bin")):

        print(f"Loading best model from: {potential_checkpoint}")
        # Load the base model again
        base_inference_model = RobertaForSequenceClassification.from_pretrained(
            base_model,
            id2label=id2label,
            num_labels=num_labels
        )
        # Load the PEFT adapter
        inference_model = PeftModel.from_pretrained(base_inference_model, potential_checkpoint)
        inference_model.merge_and_unload() # Optional: Merge adapter weights for potentially faster inference

        # Load unlabelled data (assuming it exists as in the original script)
        unlabelled_data_path = "test_unlabelled.pkl"
        if os.path.exists(unlabelled_data_path):
            print("Loading unlabelled data...")
            unlabelled_dataset_pd = pd.read_pickle(unlabelled_data_path)
            # Convert pandas DataFrame to Hugging Face Dataset
            unlabelled_dataset_hf = Dataset.from_pandas(unlabelled_dataset_pd)

            print("Preprocessing unlabelled data...")
            # Make sure to remove only 'text' if other columns exist and are needed, else remove all original columns
            columns_to_remove = [col for col in unlabelled_dataset_hf.column_names if col not in ['input_ids', 'attention_mask', 'labels']] # Adjust if needed
            test_dataset_tokenized = unlabelled_dataset_hf.map(
                preprocess,
                batched=True,
                remove_columns=columns_to_remove # Adapt based on actual columns
            )


            print("Running inference on unlabelled data...")
            preds = evaluate_model(
                inference_model,
                test_dataset_tokenized,
                labelled=False, # Set labelled to False
                batch_size=64, # Use appropriate batch size
                data_collator=data_collator
            )

            # Save predictions
            df_output = pd.DataFrame(
                {"ID": range(len(preds)), "Label": preds.cpu().numpy()} # Move preds to CPU
            )
            inference_output_path = os.path.join(output_base_dir, "best_model_inference_output.csv")
            df_output.to_csv(inference_output_path, index=False)
            print(f"Inference complete. Predictions saved to {inference_output_path}")

            del base_inference_model
            del inference_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            print(f"Unlabelled data file not found at: {unlabelled_data_path}. Skipping inference.")

    else:
        print(f"Could not find saved model checkpoint at {potential_checkpoint}. Skipping inference.")
else:
    print("No successful runs found or DSE results are empty. Skipping final inference.")
