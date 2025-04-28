# Overview

This folder contains scripts, notebooks, and datasets for conducting experiments related to resume embedding extraction, classification, visualization, and matching job descriptions with resumes.


# Setup

To set up the Python environment, use the provided requirements.txt file:
```bash
pip install -r requirements.txt
```


# File Descriptions

## Embedding Generation and Preprocessing
preprocessor.py: Utility code to preprocess resumes and job descriptions, and generate embeddings.

get_resume_embedding.py: Generates embeddings for resumes using the following transformer embedding models:  
- sentence-transformers/all-MiniLM-L6-v2  
- sentence-transformers/all-mpnet-base-v2  
- bert-base-uncased  
- bert-large-uncased  
- roberta-base  
- distilbert-base-uncased  
- albert-base-v2

get_jd_keys_embeddings.py: Generates embeddings for job descriptions using the same transformer embedding models.

## Data
resume/: Directory containing resume files.

jd_data.csv: CSV file containing structured job description data.

resume_embeddings/: Directory storing generated embeddings for resumes by embedding type.

jd_embeddings/: Directory storing generated embeddings for job descriptions by embedding type.

## Feature Extraction and Cleaning
text_cleaning_feature_extraction.ipynb: Jupyter notebook for text extraction, cleaning, and initial feature extraction from resumes.

## Classification
resume_classification.py: Script used for training and evaluating resume classifiers (LightGBM, XGBoost, Logistic Regression) using embeddings as features. Classification results, including accuracy and F1-scores, are recorded for comparison.

classification_results/: Directory containing:  
- Classification results for each embedding-model/classifier combination.  
- Detailed classification reports and performance metrics.

## Visualization
visualize_embeddings.py: Script to generate t-SNE visualizations of resume embeddings, colored by job category.

tsne_visualizations/: Directory containing generated 2D visualizations (plots) of embeddings from each embedding type.


## Matching Resumes and Job Descriptions

resume_match.py: Implements weighted cosine similarity for matching resumes to job descriptions, leveraging embeddings generated previously.

