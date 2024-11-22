import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved model, vectorizer, and transformer
count_vect = joblib.load(r'/home/ubuntu/Hackathon24/Models/Randomforest/count_vect.pkl')
transformer = joblib.load(r'/home/ubuntu/Hackathon24/Models/Randomforest/tfidf_transformer.pkl')
rfc = joblib.load(r'/home/ubuntu/Hackathon24/Models/Randomforest/random_forest_model.pkl')

# Load test CSV
test_df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Test.csv")  # CSV format: category,sub_category,crimeaditionalinfo,English Sentences

# Preprocess and vectorize test data
x_test = test_df["English Sentences"]
y_test = test_df["category"]

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

# Predict using the saved model
y_pred = rfc.predict(x_test_tfidf)

# Save predictions to a new CSV
test_df['Predicted'] = y_pred
test_df.to_csv(r"/home/ubuntu/Hackathon24/output/Randomforest/test_results.csv", index=False)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", pd.DataFrame(classification_rep).transpose())

# Visualize metrics
# 1. Pie chart for category distribution
plt.figure(figsize=(8, 8))
test_df['category'].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Category Distribution")
plt.ylabel("")
plt.savefig("category_distribution_pie_chart.png")
plt.show()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=rfc.classes_, yticklabels=rfc.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# 3. Save classification report as a CSV
classification_report_df = pd.DataFrame(classification_rep).transpose()
classification_report_df.to_csv(r"/home/ubuntu/Hackathon24/output/Randomforest/classification_report.csv", index=True)

print("Results saved: test_results.csv, classification_report.csv, confusion_matrix.png, category_distribution_pie_chart.png")
