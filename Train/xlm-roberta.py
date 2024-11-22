import pandas as pd
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset
df = pd.read_csv(r'/home/ubuntu/Hackathon24/dataset/Train_preprocessed.csv')

# Remove the first row (which contains column names)
df.columns = ['category', 'sub_category', 'crimeadditionalinfo', 'English Senetences']  # Ensure column names are correct
df = df[1:].reset_index(drop=True)  # Drop the first row and reset index

# Label Encoding
df['label'] = pd.factorize(df['category'])[0]
label_mapping = dict(enumerate(df['category'].unique()))
print("Label mapping:", label_mapping)

# Class Weights Calculation
class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['English Senetences', 'label']])
dataset = dataset.train_test_split(test_size=0.2)

# Tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Tokenization Function
def tokenize_function(examples):
    examples['English Senetences'] = [str(text) if text is not None else "" for text in examples['English Senetences']]
    return tokenizer(examples['English Senetences'], padding="max_length", truncation=True)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Model Setup
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_mapping),use_safetensors=False).to(device)

# Custom Loss Function with Class Weights
def custom_loss_fn(logits, labels):
    loss_fn = CrossEntropyLoss(weight=class_weights)
    return loss_fn(logits.view(-1, len(label_mapping)), labels.view(-1))

# Trainer Class with Custom Loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = custom_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Define a function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir=r"/home/ubuntu/Hackathon24/output/xlm-roberta_output/results",
    evaluation_strategy="steps",  # Evaluate more frequently
    eval_steps=500,  # Evaluate every 500 steps
    save_strategy="steps",  # Save model every few steps
    save_steps=500,  # Save model every 500 steps
    save_total_limit=3,  # Limit the number of saved models
    per_device_train_batch_size=16,  # Larger batch size for efficiency
    per_device_eval_batch_size=16,
    learning_rate=2e-5,  # Slightly higher learning rate for better convergence
    num_train_epochs=10,  # Reduce number of epochs to prevent overfitting
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir=r"/home/ubuntu/Hackathon24/output/xlm-roberta_output/logs",  # Enable logging
    logging_steps=500,  # Log progress every 500 steps
)

# Adding Early Stopping Callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

# Initialize Trainer with custom loss function and metrics
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,  # Added for accuracy calculation
    callbacks=[early_stopping],
)

# Train the Model
train_history = trainer.train()

# Save the Model and Tokenizer
model.save_pretrained(r"/home/ubuntu/Hackathon24/output/xlm-roberta_output/saved_model")
tokenizer.save_pretrained(r"/home/ubuntu/Hackathon24/output/xlm-roberta_output/saved_model")

# Extract Training & Validation Loss and Accuracy for Plotting
train_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
eval_loss = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
train_accuracy = [log["accuracy"] for log in trainer.state.log_history if "accuracy" in log]
eval_accuracy = [log["eval_accuracy"] for log in trainer.state.log_history if "eval_accuracy" in log]

# Handle cases where lists may be empty
epochs = range(1, max(len(train_loss), len(eval_loss)) + 1)

# Plot Loss
plt.figure(figsize=(12, 5))
if train_loss:
    plt.plot(epochs[:len(train_loss)], train_loss, label='Training Loss')
if eval_loss:
    plt.plot(epochs[:len(eval_loss)], eval_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot Accuracy
plt.figure(figsize=(12, 5))
if train_accuracy:
    plt.plot(epochs[:len(train_accuracy)], train_accuracy, label='Training Accuracy')
if eval_accuracy:
    plt.plot(epochs[:len(eval_accuracy)], eval_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
