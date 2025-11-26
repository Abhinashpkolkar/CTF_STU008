# CTF_STU008 Solution

## Approach Summary
This repository contains the solution for the "Capture the Flag" data science challenge. The goal was to act as an AI detective to identify a manipulated book, locate a fake review, and analyze the authenticity of reviews using machine learning interpretability techniques.

## Repository Structure
* **`solver.py`**: The main Python script that automates the entire workflow:
    1.  **Flag 1**: Identifies the target book using rating filters and signature hash matching.
    2.  **Flag 2**: Locates the fake review injected with a specific hash.
    3.  **Flag 3**: Trains a Logistic Regression model and uses SHAP values to identify genuine keywords.
* **`flags.txt`**: A text file containing the three discovered flags in the required format.
* **`reflection.md`**: A summary of the methodology and forensic steps taken.

## Usage
To reproduce the results, ensure `books.csv` and `reviews.csv` are in the directory and run:
```bash
python solver.py