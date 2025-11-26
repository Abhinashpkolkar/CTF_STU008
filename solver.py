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