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

base_model = "roberta-base"

dataset = load_dataset("ag_news", split="train")
tokenizer = RobertaTokenizer.from_pretrained(base_model)


def preprocess(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True)
    return tokenized


tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Extract the number of classess and their names
num_labels = dataset.features["label"].num_classes
class_names = dataset.features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
# We will need this for our classifier.
id2label = {i: label for i, label in enumerate(class_names)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

model = RobertaForSequenceClassification.from_pretrained(base_model, id2label=id2label)

# Split the original training set
split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# Configure LoRA
peft_config = LoraConfig(
    r=7,  # LoRA rank
    lora_alpha=16,  # Alpha parameter for scaling
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    target_modules=["query", "key", "value"],  # Apply LoRA to these layers
    bias="none",  # Don't train bias parameters
    task_type="SEQ_CLS",  # Specify the task type
)

peft_model = get_peft_model(model, peft_config)

print("PEFT Model")
peft_model.print_trainable_parameters()

# To track evaluation accuracy during training
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}


# Setup Training args
output_dir = "results"
training_args = TrainingArguments(
    output_dir=output_dir,
    report_to=None,
    eval_strategy="steps",
    logging_steps=100,
    learning_rate=1e-5,
    num_train_epochs=1,
    max_steps=1200,
    use_cpu=False,
    dataloader_num_workers=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    optim="adamw_torch",
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)


def get_trainer(model):
    return Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


peft_lora_finetuning_trainer = get_trainer(peft_model)

result = peft_lora_finetuning_trainer.train()

from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm


def evaluate_model(
    inference_model, dataset, labelled=True, batch_size=8, data_collator=None
):
    """
    Evaluate a PEFT model on a dataset.

    Args:
        inference_model: The model to evaluate.
        dataset: The dataset (Hugging Face Dataset) to run inference on.
        labelled (bool): If True, the dataset includes labels and metrics will be computed.
                         If False, only predictions will be returned.
        batch_size (int): Batch size for inference.
        data_collator: Function to collate batches. If None, the default collate_fn is used.

    Returns:
        If labelled is True, returns a tuple (metrics, predictions)
        If labelled is False, returns the predictions.
    """
    # Create the DataLoader
    eval_dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    if labelled:
        metric = evaluate.load("accuracy")

    # Loop over the DataLoader
    for batch in tqdm(eval_dataloader):
        # Move each tensor in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())

        if labelled:
            # Expecting that labels are provided under the "labels" key.
            references = batch["labels"]
            metric.add_batch(
                predictions=predictions.cpu().numpy(),
                references=references.cpu().numpy(),
            )

    # Concatenate predictions from all batches
    all_predictions = torch.cat(all_predictions, dim=0)

    if labelled:
        eval_metric = metric.compute()
        print("Evaluation Metric:", eval_metric)
        return eval_metric, all_predictions
    else:
        return all_predictions

    # Check evaluation accuracy


_, _ = evaluate_model(peft_model, eval_dataset, True, 8, data_collator)

# Load your unlabelled data
unlabelled_dataset = pd.read_pickle("test_unlabelled.pkl")
test_dataset = unlabelled_dataset.map(preprocess, batched=True, remove_columns=["text"])

# Run inference and save predictions
preds = evaluate_model(peft_model, test_dataset, False, 8, data_collator)
df_output = pd.DataFrame(
    {"ID": range(len(preds)), "Label": preds.numpy()}  # or preds.tolist()
)
df_output.to_csv(os.path.join(output_dir, "inference_output.csv"), index=False)
print("Inference complete. Predictions saved to inference_output.csv")
