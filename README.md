# IndiaAI-Hackathon-2024

## **Team Name**: SHAKTI  

### **Team Members**:  
1. **Rina Mishra**: PhD Scholar, Department of Computer Science and Engineering with specialization in Cybersecurity, IIT Jammu  
2. **Shreya Singh**: MTech Research Assistant, Department of Computer Science and Engineering with specialization in Computer Technology, IIT Jammu  
3. **Khushi Verma**: B.Tech Student, Department of Computer Science and Engineering, IIT Jammu  

---

## **Aim**  
To develop an NLP model that categorizes complaints based on the victim, type of fraud, and other relevant parameters using advanced text classification techniques.  

---

## **Dataset**  

### **Dataset Details**:  
- **Source**: IndiaAI CyberGuard AI Hackathon  
- **Format**: CSV file  
- **Columns**:  
  - `crimeaditionalinfo`: Additional crime-related information  
  - `Category`: Main category of crime  
  - `sub_category`: Sub-category of the crime  

---

## **Data Analysis**  

### Key Challenges and Solutions  

1. **Unbalanced Data**  
   - **Issue**: Significant class imbalance, risking biased model training.  
   - **Solution**: Oversampling minor classes using techniques like Synthetic Data Generation to balance the dataset.  
   - **Example**:  
     - Major Class: "Online Financial Fraud" (57,434 samples)  
     - Minor Class: "Ransomware" (56 samples)  

2. **Mixed Language Content**  
   - **Issue**: Data contained both English and Hinglish, leading to inconsistencies.  
   - **Solution**: Translated Hinglish to English using Google Sheets' `GoogleTranslate` function for improved accuracy.  
   - **Example (Before Translation)**:  
     ```
     RUPAY KA RECHARGE PHONE PE SE KARNE KE UPRANT THODI DER BAD ACCOUNT SE PASE KAT CHUKE THE ...
     ```  
   - **Example (After Translation)**:  
     ```
     After recharging through RuPay on the phone, money was debited from the account shortly after ...
     ```

3. **Use of Abbreviations**  
   - **Issue**: Ambiguity caused by abbreviations.  
   - **Solution**: Expanded abbreviations using a custom dictionary.  
   - **Example**:  
     - `SBI YONO`: State Bank of India You Only Need One  

4. **Noisy Data**  
   - **Issue**: Presence of URLs, punctuation, and nonsensical text.  
   - **Solution**: Applied text cleaning techniques:
     - Converted to lowercase
     - Removed URLs, punctuation, and stopwords
     - Performed lemmatization.  
     - Removed 2 length words, which was not contributing much.

5. **Mislabeled and Nonsense Data**  
   - **Issue**: Incorrect labels, NULL values, and gibberish entries.  
   - **Solution**: Removed mislabeled rows and nonsensical entries to ensure data quality.  

---

## **Development of the NLP Model**  

### **Problem Statement**  
The objective of this project is to develop an NLP model that automatically categorizes complaints based on:  
- **Victim Type**  
- **Type of Fraud**  
- **Other Relevant Parameters**  

This solution aims to enhance the efficiency and reliability of complaint handling systems by automating text classification.  

---

## **Objectives**  

1. **Text Preprocessing**:
   - Tokenization
   - Removal of stop words
   - lemmatization for normalization
   - Cleaning text by removing special characters, numbers, and other noise  

2. **Model Development**:
   - Train and evaluate various models for text classification.
   - Compare traditional ML models with advanced deep learning models like BERT.  

3. **Performance Evaluation**:
   - Use metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
   - Visualize results using confusion matrices and performance charts.  

---

## **Workflow**  

### 1. Exploratory Data Analysis (EDA)  
- Analyze data for missing values, imbalanced classes, and common patterns.  
- Visualize data distribution using `matplotlib` and `seaborn`.
- Used wordcloud to see the most impactful words in each category. 

### 2. Data Preprocessing  
- **Steps**:  
  1. Convert text to lowercase.  
  2. Tokenize text into words.  
  3. Remove stop words.  
  4. Apply stemming or lemmatization.  
  5. Remove special characters, punctuation, and numbers.  
- **Tools Used**:  
  - `NLTK`  
  - `spaCy`  
  - `re` (Regular Expressions)  

### 3. Model Development  
- **Models Trained**:  
  - Logistic Regression  
  - Naive Bayes  
  - Support Vector Machines (SVM)  
  - Random Forest  
  - BERT/Transformers  
- **Feature Extraction Methods**:  
  - Bag-of-Words (BoW)  
  - TF-IDF  
  - Word Embeddings (Word2Vec, GloVe)  
  - Pre-trained embeddings (e.g., BERT).  

### 4. Model Evaluation  
- **Metrics Used**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
- **Visualization**:  
  - Confusion matrices  
  - Bar charts comparing model performances  

### 5. Comparison of Models  

| **Model**               | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-------------------------|--------------|---------------|------------|--------------|
| Logistic Regression     | 85.0%       | 83.0%         | 84.0%      | 83.5%        |
| Naive Bayes             | 82.0%       | 80.0%         | 81.0%      | 80.5%        |
| Support Vector Machine  | 88.0%       | 86.0%         | 87.0%      | 86.5%        |
| Random Forest           | 87.0%       | 85.0%         | 86.0%      | 85.5%        |
| **BERT**                | **92.0%**   | **90.5%**     | **91.0%**  | **90.7%**    |

---

## **Folder Structure**  

```plaintext
.
├── data/
│   ├── complaints.csv         # Raw dataset
│   ├── preprocessed.csv       # Processed dataset
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis notebook
│   ├── Model_Comparison.ipynb # Model training and comparison notebook
├── scripts/
│   ├── preprocess.py          # Text preprocessing script
│   ├── train_model.py         # Model training script
│   ├── evaluate.py            # Model evaluation script
├── models/
│   ├── best_model.pkl         # Saved best-performing model
├── outputs/
│   ├── evaluation_report.txt  # Text report of evaluation metrics
│   ├── confusion_matrix.png   # Confusion matrix visualization
├── README.md                  # Project documentation
