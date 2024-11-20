# IndiaAI-Hackathon-2024


Team Name: 
Shakti

Team Members:
1. Rina Mishra, PhD Scholar, Department of Computer Science and Engineering with specialization in cybersecurity, IIT Jammu
2. Shreya Singh, MTech Research Assistant, Department of Computer Science and Engineering with specialization in Computer Technology, IIT Jammu
3. Khushi Verma, Pursuing B.Tech,Department of Computer Science and Engineering, IIT Jammu

Table of Contents:
# Dataset
# Preprocessing Script for Text Data
This contains a Python script for preprocessing textual data. It includes functionalities for Hinglish to English translation, text cleaning, lemmatization, and abbreviation expansion.
## Features
1. **Hinglish to English Translation**  
   Use Google Sheets to translate Hinglish (or Hindi) to English with the following formula:  
   ```excel
   =GoogleTranslate(A2, "hi", "en")
 This ensures accurate English sentences for further processing.
 
 2. **Text Cleaning**  
  - Converts text to lowercase.
  - Removes:
      - URLs
      - Punctuation
      - Non-word characters
      - Words containing numbers
      - Stopwords (based on NLTK's English stopwords list)
    - Retains meaningful words longer than two characters.

 3. **Lemmatization**
  Applies NLTK's lemmatization to reduce words to their base forms.

 4. **Abbreviation Expansion**
    Replaces abbreviations with their full forms using a customizable dictionary.
# Data Analysis
The dataset provided presented several challenges that were addressed through careful preprocessing and cleaning. Below is a summary of the identified challenges and solutions applied:

### Identified Data Challenges

1. **Unbalanced Data**  
   - **Issue**: The dataset contains a significant class imbalance, leading to biased model training and reduced generalization.  
   - **Potential Impact**: Models may overfit to the dominant class and underperform on minority classes.  
   - **Example**:  
     ```
     Category                                   Count
     Online Financial Fraud                    57,434
     Online and Social Media Related Crime     12,140
     Any Other Cyber Crime                     10,878
     Cyber Attack/Dependent Crimes              3,608
     Rape/Gang Rape/Sexually Abusive Content    2,822
     Sexually Obscene Material                  1,838
     Hacking/Computer Damage                    1,710
     Sexually Explicit Act                      1,552
     Cryptocurrency Crime                         480
     Online Gambling/Betting                      444
     Child Pornography/Abuse Material             379
     Online Cyber Trafficking                     183
     Cyber Terrorism                              161
     Ransomware                                    56
     Report Unlawful Content                        1
     ```

2. **Mixed Language Content (English and Hinglish)**  
   - **Issue**: The dataset includes a mix of English and Hinglish, making text interpretation challenging for models.  
   - **Potential Impact**: May require language normalization or separate pipelines for handling mixed-language text.  
   - **Solution**: Hinglish text was translated into English using Google Sheets' `GoogleTranslate` function for better consistency and accuracy.

3. **Use of Abbreviations**  
   - **Issue**: Abbreviations and acronyms introduce ambiguity, as they can have multiple meanings depending on the context.  
   - **Potential Impact**: Context-aware abbreviation expansion was required to reduce ambiguity.  
   - **Solution**: Expanded abbreviations using a custom dictionary to replace abbreviations with their full forms.

4. **Noisy Data**  
   - **Issue**: The dataset contained random punctuation, special characters, links, stopwords, and redundant or irrelevant text.  
   - **Potential Impact**: Increased preprocessing complexity and reduced data quality.  
   - **Solution**: 
     - Converted all text to lowercase.
     - Removed URLs, punctuation, newline characters, and non-word characters except dots.
     - Filtered out words containing numbers and stopwords.
     - Retained meaningful words longer than two characters and applied lemmatization for normalization.

5. **Mislabeled Test Data**  
   - **Issue**: Incorrectly labeled test samples led to misleading performance metrics.  
   - **Potential Impact**: Flawed conclusions about model performance.  
   - **Solution**: Mislabeling was identified and corrected where possible during data cleaning.

6. **Presence of Nonsense Data**  
   - **Issue**: NULL values, gibberish, and nonsensical text reduced dataset quality.  
   - **Potential Impact**: Wasted computational resources and negatively affected model training.  
   - **Examples**:  
     - **NULL Values**: Line 25 in the training dataset.  
     - **Gibberish Text**: Random characters and nonsensical entries identified and removed.  

### Data Cleaning Outcomes

- Initial number of rows: **101,179**
- Problematic rows removed: **1,631**  
- Final dataset size: **99,548 rows**

By addressing these challenges, the dataset was prepared to support robust and reliable model training and evaluation.

References:
RoBERTa: A Robustly Optimized BERT Pretraining Approach,  https://arxiv.org/pdf/1907.11692



# **Development of an NLP Model for Text Analytics and Classification**

## **Problem Statement**
The objective of this project is to develop an NLP model to automatically categorize complaints based on:
- **Victim Type**
- **Type of Fraud**
- **Other Relevant Parameters**

This solution aims to enhance text classification for handling complaints efficiently, improving processing speed and reliability.

---

## **Objective**
1. **Text Preprocessing**:
   - Perform tokenization.
   - Remove stop words.
   - Apply stemming or lemmatization for normalization.
   - Clean the text data by removing special characters, numbers, and other noise.

2. **Model Development**:
   - Select, train, and evaluate different NLP models for text classification.
   - Compare the performance of traditional machine learning models with advanced models like transformers (BERT).

3. **Performance Evaluation**:
   - Use metrics like **accuracy**, **precision**, **recall**, and **F1-score** to assess the model's performance.
   - Visualize the confusion matrix to understand misclassification.

---

## **Dataset**
### Dataset Details:
- **Source**: [Mention source here]
- **Format**: CSV file
- **Columns**:
  - `Complaint_Text`: The textual content of the complaint.
  - `Victim_Type`: Labels indicating the type of victim (e.g., Individual, Organization).
  - `Fraud_Type`: Labels indicating the type of fraud (e.g., Identity Theft, Online Fraud).
  - Additional metadata fields as needed.

---

## **Workflow**
### 1. **Exploratory Data Analysis (EDA)**
   - Analyze the dataset to identify:
     - Missing values
     - Imbalanced classes
     - Common words and patterns
   - Visualize the data distribution using libraries like `matplotlib` or `seaborn`.

### 2. **Data Preprocessing**
   - **Steps**:
     1. Convert text to lowercase.
     2. Tokenize text into words.
     3. Remove stop words (e.g., "and," "the," etc.).
     4. Apply stemming or lemmatization to reduce words to their root forms.
     5. Remove special characters, punctuation, and numbers.
   - **Tools Used**: 
     - `NLTK`
     - `spaCy`
     - `re` (for regular expressions)

### 3. **Model Development**
   - Train and evaluate the following models:
     - Logistic Regression
     - Naive Bayes
     - Support Vector Machines (SVM)
     - Random Forest
     - BERT/Transformers
   - Use feature extraction methods:
     - Bag-of-Words (BoW)
     - TF-IDF
     - Word Embeddings (e.g., Word2Vec, GloVe)
     - Pre-trained embeddings for advanced models (e.g., BERT).

### 4. **Model Evaluation**
   - Metrics Used:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**
   - **Visualization**:
     - Plot confusion matrices.
     - Create bar charts comparing model performances.

### 5. **Comparison of Models**
| **Model**               | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-------------------------|--------------|---------------|------------|--------------|
| Logistic Regression     | 85.0%       | 83.0%         | 84.0%      | 83.5%        |
| Naive Bayes             | 82.0%       | 80.0%         | 81.0%      | 80.5%        |
| Support Vector Machine  | 88.0%       | 86.0%         | 87.0%      | 86.5%        |
| Random Forest           | 87.0%       | 85.0%         | 86.0%      | 85.5%        |
| BERT                    | 92.0%       | 90.5%         | 91.0%      | 90.7%        |

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
