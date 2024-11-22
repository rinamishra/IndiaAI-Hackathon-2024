# Import necessary libraries
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm for progress bar

# Load dataset
df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Train_preprocessed.csv", engine='python', encoding='UTF-8')

# Check the distribution of categories
print(df['category'].value_counts())

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    df["English Sentences"], df["category"], test_size=0.25, random_state=42
)

x_train = x_train.fillna("").astype(str)
x_test = x_test.fillna("").astype(str)

# Apply CountVectorizer and TfidfTransformer
count_vect = CountVectorizer(ngram_range=(1, 2))        
transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

# Save the vectorizer
joblib.dump(count_vect, r'/home/ubuntu/Hackathon24/Models/Multinominal_NB/count_vect.pkl')
joblib.dump(transformer, r'/home/ubuntu/Hackathon24/Models/Multinominal_NB/tfidf_transformer.pkl')

# Train a Naive Bayes model with progress bar
mnb = MultinomialNB()

# Display progress bar for training
print("Training Multinomial Naive Bayes...")
for _ in tqdm(range(1), desc="Training Model", ncols=100):
    mnb.fit(x_train_tfidf, y_train)

# Make predictions on the test set
y_pred = mnb.predict(x_test_tfidf)

# Calculate accuracy and print classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(mnb, r'/home/ubuntu/Hackathon24/Models/Multinominal_NB/Text_NB_Model.pkl')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['category'].unique(), yticklabels=df['category'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation with progress bar
print("Performing Cross-validation...")
cv_scores = []
for score in tqdm(cross_val_score(mnb, x_train_tfidf, y_train, cv=10), desc="Cross-validation", ncols=100):
    cv_scores.append(score)
    
print("Cross-validated scores:", cv_scores)
