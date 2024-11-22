import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm

# Load the dataset
df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Train_preprocessed.csv")

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    df["English Sentences"], df["category"], test_size=0.25, random_state=42
)

# Preprocessing with CountVectorizer and TfidfTransformer
count_vect = CountVectorizer(ngram_range=(1, 2))
transformer = TfidfTransformer(norm="l2", sublinear_tf=True)

# Replace NaN values with an empty string and ensure all values are strings
x_train = x_train.fillna("").astype(str)
x_test = x_test.fillna("").astype(str)

# Fit and transform the data
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

# Save vectorizers
joblib.dump(count_vect, r"/home/ubuntu/Hackathon24/Models/LR/count_vect.pkl")
joblib.dump(transformer, r"/home/ubuntu/Hackathon24/Models/LR/tfidf_transformer.pkl")

# Train Logistic Regression model with progress bar
lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)

# Use tqdm to show a progress bar during fitting (if desired)
for _ in tqdm(range(1), desc="Training Logistic Regression", ncols=100):
    lr.fit(x_train_tfidf, y_train)

# Save model
joblib.dump(lr, r"/home/ubuntu/Hackathon24/Models/LR/Text_LR.pkl")

# Evaluate the model
y_pred = lr.predict(x_test_tfidf)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Cross-validation with progress bar
print("Performing Cross-validation...")
cv_scores = []
for score in tqdm(cross_val_score(lr, x_train_tfidf, y_train, cv=10), desc="Cross-validation", ncols=100):
    cv_scores.append(score)
    
print("Cross-validated scores:", cv_scores)
