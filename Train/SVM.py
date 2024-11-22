# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Train_preprocessed.csv", engine="python", encoding="UTF-8")

# Splitting the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(
    df["English Sentences"], df["category"], test_size=0.25, random_state=42
)

# Applying N-grams and TF-IDF transformation
count_vect = CountVectorizer(ngram_range=(1, 2))
transformer = TfidfTransformer(norm="l2", sublinear_tf=True)

# Replace NaN values with an empty string and ensure all values are strings
x_train = x_train.fillna("").astype(str)
x_test = x_test.fillna("").astype(str)


x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

# Save vectorizer and transformer
joblib.dump(count_vect, r'/home/ubuntu/Hackathon24/Models/SVM/count_vect.pkl')
joblib.dump(transformer, r'/home/ubuntu/Hackathon24/Models/SVM/tfidf_transformer.pkl')

# Train the SVM model
svc = LinearSVC()
svc.fit(x_train_tfidf, y_train)

# Save the model
joblib.dump(svc, r'/home/ubuntu/Hackathon24/Models/SVM/Text_SVM.pkl')

# Evaluate the model
y_pred = svc.predict(x_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Cross-validation scores
scores = cross_val_score(svc, x_train_tfidf, y_train, cv=10)
print("Cross-validated scores:", scores)
