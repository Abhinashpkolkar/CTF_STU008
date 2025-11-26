# Reflection on CTF Methodology

To solve this challenge, I adopted a structured data forensic approach combining cryptographic hashing, data filtering, and machine learning interpretability.

**Step 1: Identification (Flag 1)**
I began by computing the SHA256 hash of my student ID ("STU008") to generate a unique signature. I then filtered the `books.csv` dataset to isolate books with exactly 1234 ratings and a perfect 5.0 average. By cross-referencing these candidates with `reviews.csv`, I searched for my signature hash within the review text. A key hurdle was handling case sensitivity, as the injected hash was lowercase while my computed hash was uppercase.

**Step 2: The Fake Review (Flag 2)**
Upon finding the review containing the hash, I extracted the fake review's text. I noticed a data anomaly where this specific review lacked a `parent_asin` but had a valid `asin`. This required adjusting my data pipeline to check both columns to ensure the fake review was included in subsequent analyses.

**Step 3: Authenticity Analysis (Flag 3)**
To differentiate between genuine and suspicious reviews, I engineered a labeling function based on the provided heuristics (length, rating, and superlatives). I trained a Logistic Regression model using TF-IDF vectorization on the labeled data. Finally, I employed SHAP (SHapley Additive exPlanations) to interpret the model's decision-boundary. By averaging the SHAP values for the "Genuine" class, I identified the top three words that most strongly indicated authenticity, which formed the final flag.