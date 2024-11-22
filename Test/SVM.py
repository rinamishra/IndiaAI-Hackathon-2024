# Import necessary libraries
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved vectorizer, transformer, and model
count_vect = joblib.load(r'/home/ubuntu/Hackathon24/Models/SVM/count_vect.pkl')
transformer = joblib.load(r'/home/ubuntu/Hackathon24/Models/SVM/tfidf_transformer.pkl')
svc = joblib.load(r'/home/ubuntu/Hackathon24/Models/SVM/Text_SVM.pkl')

# Load the new dataset for testing
test_df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Test.csv")

# Extract English sentences and category labels
x_test = test_df["English Sentences"]
y_true = test_df["category"]

# Transform the test data
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

# Make predictions
y_pred = svc.predict(x_test_tfidf)

# Add predictions to the dataframe
test_df["Predicted_Category"] = y_pred

# Save predictions to a new CSV file
test_df.to_csv("SVM_Predictions.csv", index=False)

# Evaluate the model
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_true, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=svc.classes_)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=svc.classes_, yticklabels=svc.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Pie chart for distribution of predictions
plt.figure(figsize=(8, 8))
test_df["Predicted_Category"].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=90)
plt.title("Predicted Category Distribution")
plt.ylabel("")
plt.show()

# Save evaluation metrics to a CSV file
evaluation_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
evaluation_df.to_csv("SVM_Evaluation_Metrics.csv")
