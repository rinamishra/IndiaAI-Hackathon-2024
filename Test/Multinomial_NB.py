# Import necessary libraries
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and vectorizer
count_vect = joblib.load(r'/home/ubuntu/Hackathon24/Models/Multinominal_NB/count_vect.pkl')
transformer = joblib.load(r'/home/ubuntu/Hackathon24/Models/Multinominal_NB/tfidf_transformer.pkl')
mnb = joblib.load(r'/home/ubuntu/Hackathon24/Models/Multinominal_NB/Text_NB_Model.pkl')

# Load the test dataset
test_df = pd.read_csv(r"/home/ubuntu/Hackathon24/dataset/Test.csv", engine='python', encoding='UTF-8')

# Apply the same vectorization on the "English Sentences" column
x_test_counts = count_vect.transform(test_df["English Sentences"])
x_test_tfidf = transformer.transform(x_test_counts)

# Make predictions using the saved model
y_pred = mnb.predict(x_test_tfidf)

# Store predictions in the test dataframe
test_df['Predicted Category'] = y_pred

# Save the new dataframe with predictions to a CSV file
test_df.to_csv('predicted_test_data.csv', index=False)

# Calculate accuracy and print classification report
accuracy = accuracy_score(test_df['category'], y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(test_df['category'], y_pred))

# Confusion Matrix
cm = confusion_matrix(test_df['category'], y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_df['category'].unique(), yticklabels=test_df['category'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Visualize the category distribution of the predicted labels using a pie chart
category_counts = test_df['Predicted Category'].value_counts()
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Predicted Category Distribution')
plt.ylabel('')
plt.show()

# Visualizing the accuracy across categories using a bar chart
category_accuracy = test_df.groupby('Predicted Category').apply(lambda x: accuracy_score(x['category'], x['Predicted Category']))
category_accuracy.plot(kind='bar')
plt.title('Accuracy per Category')
plt.xlabel('Category')
plt.ylabel('Accuracy')
plt.show()

# Save accuracy and classification metrics to a text file for analysis
with open('/home/ubuntu/Hackathon24/output/Multinominal_NB/model_performance.txt', 'w') as f:
    f.write("Accuracy: {}\n".format(accuracy))
    f.write("Classification Report:\n")
    f.write(str(classification_report(test_df['category'], y_pred)))

