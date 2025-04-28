# NLP Capstone Project: Resume-JD Matching System

## Overview

This project implements a resume-to-job matching system using various NLP techniques. It extracts structured information from resumes and job descriptions (JDs), embeds them using sentence-transformers, and computes similarity scores to find the best job fits.

---

## Project Structure

```
nlp-capstone-project/
├── exp/                  # Experiment logs, notebooks, results
├── jd_embeddings/        # Precomputed job description embeddings (CSV)
├── resume/               # Resume PDFs used for testing
├── resume-matcher-gui/   # Web-based Graphical User Interface code
├── src/                  # Main source code for preprocessing, embedding, matching, and API
│   ├── main.py           # Command-line flow for processing and matching a resume
│   ├── api.py            # FastAPI server to upload resumes and return matching JDs
│   ├── preprocessor.py   # Text cleaning, lemmatization, and embedding utilities
│   └── ...               # Additional logic modules
├── .gitattributes        # Git LFS tracking for large files (e.g., PDFs, embeddings)
├── .DS_Store             # (macOS metadata file, can be ignored)
└── requirements.txt      # Python dependencies
```

---

## ⚙Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Atishaysjain/nlp-capstone-project.git
cd nlp-capstone-project
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK and spaCy Assets

```bash
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

---

## How to Run?

### From Command Line

Run the main script on a sample resume:

```bash
cd src
python main.py
```

This will:
- Parse the resume
- Generate section-wise embeddings
- Match against precomputed JD embeddings
- Print and save top job matches

> Make sure `resume/` contains the test resume and `jd_embeddings/` has the preprocessed JD vectors.

---

### Start the API Server

Run the FastAPI app:

```bash
cd src
uvicorn api:app --reload
```

Open in browser:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)

#### POST `/get_matches/`

**Description**: Upload a resume PDF and get the top matching job descriptions.  
**Payload**: `multipart/form-data` with a single `file` field.

---

## Evaluation Metrics

When evaluating prediction quality using ground truth JD matches, the system reports:

- **Top-1 Accuracy** – how often the top JD is relevant
- **Top-5 GT Covered in Top-25** – recall metric
- **Mean Reciprocal Rank (MRR)** – effectiveness of ranked prediction list

These are computed in the `evaluate_predictions()` function in `main.py`.

---

## Notes on Folder Contents

- `exp/`: Contains experiment results, logs, and variations of runs (for model tuning or evaluation)
- `resume/`: Stores test resumes (PDFs) that are passed through the pipeline
- `jd_embeddings/`: Stores JD vectors that were generated ahead of time and used during matching

---

## Dependencies

See `requirements.txt`. Key libraries include:

- `fastapi`, `uvicorn`
- `sentence-transformers`, `torch`, `scikit-learn`
- `spacy`, `nltk`, `pandas`, `numpy`, `PyMuPDF`

---
