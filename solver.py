# Importing libraries.
import pandas as pd
import numpy as np
import hashlib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import shap

#Input data.
student_id = "STU008"
hash_full = hashlib.sha256(student_id.encode()).hexdigest()
your_hash = hash_full[:8].upper()
print(f"Hash: {your_hash}")

# Load both CSV files
df_books = pd.read_csv('books.csv')
df_reviews = pd.read_csv('reviews.csv')

# Finding review and extracting the asin.
review_match = df_reviews[df_reviews['text'].str.contains(your_hash, na=False, case=False)]

# Take the matching row
target_row = review_match.iloc[0]

target_asin = target_row['asin']
review_title = target_row['title']

# Match reviews['asin'] with books['parent_asin']
book_match = df_books[df_books['parent_asin'] == target_asin]

book_title = book_match.iloc[0]['title']
print(f"Book Title: {book_title}")

# Computing Flag1

# Removing spaces from the title.
title_no_spaces = "".join(book_title.split())

# Extracting first 8 chars
flag1_input = title_no_spaces[:8]

flag1 = hashlib.sha256(flag1_input.encode()).hexdigest()
print(f"FLAG1: {flag1}")

# Computing Flag2( The fake review has already been located in the aove cells)
flag2 = your_hash
fake_review_row = review_match.iloc[0]
fake_review_text = fake_review_row['text']
print(f"Fake review : {fake_review_text}")  
print(f" FLAG2: {flag2}")

# Computing Flag 3

# Use the specific asin value we found initially.
target_asin = '0007144350'

# Extracted from STU008
numeric_id = "008"

# Data preparation.

# Filtering the reviews.
mask = (df_reviews['parent_asin'] == target_asin) | (df_reviews['asin'] == target_asin)
book_reviews = df_reviews[mask].copy()
book_reviews['text'] = book_reviews['text'].fillna('')

# Logic for labelling.
superlatives = ['best', 'amazing', 'perfect', 'awesome', 'excellent', 'masterpiece', 'wonderful', 'incredible']

def label_review(row):
    text = str(row['text']).lower()
    word_count = len(text.split())
    
    # Suspicious: 5star + short(less than 15 words) + superlatives
    if row['rating'] == 5.0 and word_count < 15 and any(s in text for s in superlatives):
        return 1 
    # Genuine: 5-star + detailed (more than 40 words)
    elif row['rating'] == 5.0 and word_count > 40:
        return 0 
    return -1

# Applying the labeling and filter
book_reviews['label'] = book_reviews.apply(label_review, axis=1)
train_df = book_reviews[book_reviews['label'] != -1].copy()    

# Model training.
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(train_df['text'])
y = train_df['label']

model = LogisticRegression()
model.fit(X, y)

# SHAP Analysis
# Analyze genuine vs suspicious.
explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
shap_values = explainer.shap_values(X)
feature_names = vectorizer.get_feature_names_out()

# Identifying top 3 words reducing suspicion (most negative SHAP values for class 1).
mean_shap = np.mean(shap_values, axis=0)
top_indices = np.argsort(mean_shap)[:3] 
top_words = [feature_names[i] for i in top_indices]

print(f"Top 3 words that reduce suspicion: {top_words}")

# Generating FLAG 3.

# Concatenate words + numeric ID -> SHA256 -> First 10 chars 
combined_string = "".join(top_words) + numeric_id
flag3_hash = hashlib.sha256(combined_string.encode()).hexdigest()
flag3_final = f"FLAG3{{{flag3_hash[:10]}}}"
print(f"FLAG3: {flag3_final}")
