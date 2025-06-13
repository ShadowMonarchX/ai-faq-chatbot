import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch

# Load and prepare data
df = pd.read_csv("../../data/faq_dataset.csv")
df = df[['question', 'answer']].dropna()
df['label'] = df['answer'].astype('category').cat.codes  # convert answer to label

label2text = dict(enumerate(df['answer'].astype('category').cat.categories))
text2label = {v: k for k, v in label2text.items()}

df = df[['question', 'label']]
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Load tokenizer and model
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch["question"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(set(df['label']))
)

# Training config
training_args = TrainingArguments(
    output_dir="models/faq-model",
    eval_strategy="epoch", # Changed from evaluation_strategy
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=10,
    save_total_limit=2,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("models/faq-model")
    tokenizer.save_pretrained("models/faq-model")
    print("Fine-tuning complete and model saved at models/faq-model")

    # Optional: save label mapping
    with open("models/faq-model/label_map.txt", "w") as f:
        for label_id, text in label2text.items():
            f.write(f"{label_id}:{text}\n")
