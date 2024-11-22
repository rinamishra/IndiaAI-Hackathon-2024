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
| Logistic Regression     | 77.0%       | 74.0%         | 77.0%      | 73.0%        |
| Naive Bayes             | 69.0%       | 67.0%         | 69.0%      | 59.0%        |
| Support Vector Machine  | 76.0%       | 74.0%         | 76.0%      | 74.0%        |
| Random Forest           | 54.43%      | 72.59%        | 54.43%     | 60.54%       |
| BERT               | 76.04%      | 73.00%        |74.00%      | 74.00%       |

---

## **Folder Structure**  

```plaintext
.
├── dataset/                    
│   ├── data.csv               # Here, we have added a placeholder for the Google Drive link that contains all the data.
├── Train/
│   ├── train.py               # Training scripts of all the models
├── Test/
│   ├── Test.py                # Testing scripts of all the models
├── Models/
│   ├── model.pkl              # Here, we have added a placeholder for the Google Drive link that contains all the models.
├── Submodels/
│   ├── sub_model.pkl          # Here, we have added a placeholder for the Google Drive link that contains all the Submodels.
├── Results/
│   ├── result/txt             # Result of all the models
├── README.md                  # Project documentation
├── main.py                    # Main pipeline for predicting category and subcategories of crime 

The "Models" folder does not contain the actual models due to size limitations. However, we have provided a link to a Google Drive folder that contains all the models. You can access the models via the following link: (https://drive.google.com/drive/folders/12SHU7rQauIodFaSyHYUnhsHLD9iLzD2A).And the link for submodels is (https://drive.google.com/drive/folders/1-JWpL3xzudw0rQq2VlywmgiFvuxnBYO9) .
Dataset folder contains a readme file that contains link to drive that has all datset ( https://drive.google.com/drive/folders/1ksuRSVLCyuTiRRI0hE9GxrBpfwMu_AOT)

```

## **References**  
1.  A Complete Process of Text Classification System Using State-of-the-Art NLP Models
Varun Dogra, 2022.
2. Cortes, C., & Vapnik, V. (1995). "Support-Vector Networks." Machine Learning, 20, 273–297.
3. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
4. Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5–32.
5. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT 2019.
6. Conneau, A., et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
7. Language Classification: https://github.com/pemistahl/lingua-go
8. Language Translation: https://github.com/hazzillrodriguez/flask-multi-language
9.Gupta, P., & Joshi, S. (2018). "Handling Code-Mixed Hindi-English Social Media Data: A Language Identification Perspective." Proceedings of the ACL.

