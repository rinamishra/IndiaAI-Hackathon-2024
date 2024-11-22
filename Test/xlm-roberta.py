import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
from sklearn.metrics import classification_report, accuracy_score

# Paths
test_csv_path = r'/home/ubuntu/Hackathon24/dataset/Test.csv'
model_path = r'/home/ubuntu/Hackathon24/Models/xlm-roberta_output/saved_model'
output_csv_path = r'/home/ubuntu/Hackathon24/output/xlm-roberta_output/test_results.csv'
metrics_txt_path = r'/home/ubuntu/Hackathon24/output/xlm-roberta_output/metrics_summary.txt'

# Load the trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

# Enable gradient checkpointing for memory efficiency (optional)
model.gradient_checkpointing_enable()

# Load the test dataset
test_df = pd.read_csv(test_csv_path)
test_df.columns = ['category', 'sub_category', 'crimeadditionalinfo', 'English Sentences']  # Ensure columns are correctly named
test_df = test_df[1:].reset_index(drop=True)  # Drop the first row if it contains column names

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(test_df['category'].unique())}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
test_df['label'] = test_df['category'].map(label_mapping)

# Tokenize the test sentences
def tokenize_texts(texts, batch_size=1):  # Reduced batch size to save memory
    texts = texts.apply(lambda x: str(x) if pd.notna(x) else "")
    tokenized = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        tokenized_batch = tokenizer(batch_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=256).to(device)
        tokenized.append(tokenized_batch)
    return tokenized

tokenized_texts = tokenize_texts(test_df['English Sentences'].fillna(""))

# Predict
model.eval()
all_predictions = []
with torch.no_grad():
    for batch in tokenized_texts:
        with torch.cuda.amp.autocast():  # Enable mixed precision inference
            outputs = model(**batch)  # Forward pass
        predictions = torch.argmax(outputs.logits, axis=1).cpu().numpy()
        all_predictions.extend(predictions)  # Store predictions
        torch.cuda.empty_cache()  # Clear unused GPU memory

# Add predictions to the dataframe
test_df['predicted_label'] = all_predictions
test_df['predicted_category'] = test_df['predicted_label'].map(reverse_label_mapping)

# Calculate Metrics
accuracy = accuracy_score(test_df['label'], test_df['predicted_label'])

# Get the unique labels for classification_report
unique_labels = list(label_mapping.keys())  # Use the actual unique labels from 'category'

classification_metrics = classification_report(
    test_df['label'], 
    test_df['predicted_label'], 
    target_names=unique_labels,
    labels=range(len(unique_labels))
)

# Save results to CSV
test_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")

# Save metrics to text file
with open(metrics_txt_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_metrics)

print(f"Metrics saved to: {metrics_txt_path}")
