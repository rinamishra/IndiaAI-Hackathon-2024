import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load main category models
main_count_vect = joblib.load(r"/home/ubuntu/Hackathon24/Models/LR/count_vect.pkl")
main_lr_model = joblib.load(r"/home/ubuntu/Hackathon24/Models/LR/Text_LR.pkl")

# Load subcategory models for categories with multiple subcategories (only CountVectorizer now)
sub_model_file_paths = {
    'Cyber Attack/Dependent Crimes': {
        'count_vect': '/home/ubuntu/Hackathon24/SubModel/Cyber Attack Dependent Crimes_LR/count_vect_LR.pkl',
        'model': '/home/ubuntu/Hackathon24/SubModel/Cyber Attack Dependent Crimes_LR/Text_LR.pkl',
    },
    'Hacking Damage to computer system etc': {
        'count_vect': '/home/ubuntu/Hackathon24/SubModel/Hacking_LR/count_vect_LR.pkl',
        'model': '/home/ubuntu/Hackathon24/SubModel/Hacking_LR/Text_LR.pkl',
    },
    'Online Financial Fraud': {
        'count_vect': '/home/ubuntu/Hackathon24/SubModel/Online_Financial_Fraud_LR/count_vect_LR.pkl',
        'model': '/home/ubuntu/Hackathon24/SubModel/Online_Financial_Fraud_LR/Text_LR.pkl',
    },
    'Online and Social Media Related Crime': {
        'count_vect': '/home/ubuntu/Hackathon24/SubModel/Online_social_media_and_related_crime_LR/count_vect_LR.pkl',
        'model': '/home/ubuntu/Hackathon24/SubModel/Online_social_media_and_related_crime_LR/Text_LRpkl',
    },
}

# Subcategories that don't require a model (single subcategory)
fixed_subcategories = {
    'Any Other Cyber Crime': 'Other',
    'Crime Against Women & Children': 'Computer Generated CSAM/CSEM',
    'Cryptocurrency Crime': 'Cryptocurrency Fraud',
    'Cyber Terrorism': 'Cyber Terrorism',
    'Online Cyber Trafficking': 'Online Trafficking',
    'Online Gambling Betting': 'Online Gambling Betting',
    'Ransomware': 'Ransomware',
}

# Function to predict main category and subcategory for a single example
def predict_category_and_subcategory(sentence):
    # Preprocess the input for the main category model using CountVectorizer
    X_example_counts = main_count_vect.transform([sentence])
    
    # Predict main category
    main_category = main_lr_model.predict(X_example_counts)[0]
    
    # Print the main category prediction
    print(f"Main Category Prediction: {main_category}")
    
    # If the category has a fixed subcategory, return the fixed subcategory
    if main_category in fixed_subcategories:
        subcategory_prediction = fixed_subcategories[main_category]
        print(f"Subcategory Prediction: {subcategory_prediction}")
    else:
        # For categories with multiple subcategories, load the respective model
        sub_model_paths = sub_model_file_paths.get(main_category)
        if sub_model_paths:
            sub_count_vect = joblib.load(sub_model_paths['count_vect'])
            sub_model = joblib.load(sub_model_paths['model'])
            
            # Preprocess the input for the subcategory model using CountVectorizer
            X_example_sub_counts = sub_count_vect.transform([sentence])
            
            # Predict subcategory
            subcategory_prediction = sub_model.predict(X_example_sub_counts)[0]
            print(f"Subcategory Prediction: {subcategory_prediction}")
        else:
            subcategory_prediction = "Unknown"
            print(f"Subcategory Prediction: {subcategory_prediction}")
    
    return main_category, subcategory_prediction


# Example usage
example_sentence = "kotak mahindra bank fraud"
main_category, subcategory = predict_category_and_subcategory(example_sentence)
