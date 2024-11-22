import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
from sklearn.utils import class_weight

# Load dataset
df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Train_preprocessed.csv")
print("Dataset Loaded. Sample distribution:\n", df['category'].value_counts())

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(df["English Sentences"], df["category"], test_size=0.25, random_state=42)

x_train = x_train.fillna("").astype(str)
x_test = x_test.fillna("").astype(str)

# Text vectorization with N-grams and TF-IDF
count_vect = CountVectorizer(ngram_range=(1, 2), max_features=50000)
transformer = TfidfTransformer(norm='l2', sublinear_tf=True)

# Transform training data
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

# Transform test data
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

print("Vectorization complete. Training shape:", x_train_tfidf.shape)

# Save vectorizer and transformer
joblib.dump(count_vect, r'/home/ubuntu/Hackathon24/Models/Randomforest/count_vect.pkl')
joblib.dump(transformer, r'/home/ubuntu/Hackathon24/Models/Randomforest/tfidf_transformer.pkl')

# Random Forest Classifier with a progress bar
rfc = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)
print("Training Random Forest Classifier...")
rfc.fit(x_train_tfidf, y_train)


# Save model
joblib.dump(rfc, r'/home/ubuntu/Hackathon24/Models/Randomforest/random_forest_model.pkl')

# Evaluate on test set
y_pred = rfc.predict(x_test_tfidf)
print("\nAccuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
