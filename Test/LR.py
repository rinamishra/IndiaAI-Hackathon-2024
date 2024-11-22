import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved models
count_vect = joblib.load(r"/home/ubuntu/Hackathon24/Models/LR/count_vect.pkl")
transformer = joblib.load(r"/home/ubuntu/Hackathon24/Models/LR/tfidf_transformer.pkl")
lr = joblib.load(r"/home/ubuntu/Hackathon24/Models/LR/Text_LR.pkl")

# Load the test CSV
test_df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Test.csv")

# Extract features and labels
X_test = test_df["English Sentences"]
y_true = test_df["category"]

# Preprocess the test data
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = transformer.transform(X_test_counts)

# Predict
y_pred = lr.predict(X_test_tfidf)

# Add predictions to the DataFrame
test_df["Predicted_Label"] = y_pred

# Save the results
test_df.to_csv("predicted_results.csv", index=False)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy: ", accuracy)
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=lr.classes_)
cm_df = pd.DataFrame(cm, index=lr.classes_, columns=lr.classes_)

# Visualization: Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.show()

# Visualization: Category Distribution
plt.figure(figsize=(8, 6))
test_df["Predicted_Label"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Category Distribution of Predictions")
plt.ylabel("")
plt.savefig("category_distribution.png")
plt.show()

# Save metrics as text
with open(r"/home/ubuntu/Hackathon24/output/LR/metrics_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_true, y_pred))
