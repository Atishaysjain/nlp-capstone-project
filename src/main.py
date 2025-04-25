import os
import pandas as pd
import re
import fitz  # PyMuPDF
import spacy
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from preprocessor import EncoderWithChunks
import torch
import json
import matplotlib.pyplot as plt

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

#define known headers
SECTION_HEADERS = {
    "education": ["education", "academic background", "academic qualifications", "education and training", "education details"],
    "experience": ["experience", "professional experience", "work experience", "employment history"],
    "skills": ["skills", "technical skills", "key skills", "core competencies"],
    "projects": ["projects", "personal projects"],
    "certifications": ["certifications", "licenses"],
    "summary": ["summary", "profile", "objective", "career focus"],
    "interests": ["interests", "hobbies", "interest"],
    "publications": ["publications", "publication"],
}

def preprocess(text):
    if not text:
        return ""
    
    #remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()

    #lowercase, clean special characters and whitespace
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()

    #lemmatization & stopword removal
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.lemma_.lower() not in stop_words and len(token.lemma_) > 1
    ]

    return " ".join(tokens)

def detect_sections(lines, section_headers):
    current_section = None
    sections = defaultdict(list)

    for line in lines:
        line_clean = line.lower().strip()
        matched = False
        for section, headers in section_headers.items():
            if any(re.fullmatch(rf"{h}\s*:?", line_clean) for h in headers):
                current_section = section
                matched = True
                break

        if current_section and not matched:
            sections[current_section].append(line)

    return {sec: "\n".join(content) for sec, content in sections.items()}

#extract text from the PDF resume
def extract_resume_sections(pdf_path):
    print("Extracting sections from: ",pdf_path)

    #extract raw text from PDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = detect_sections(lines, SECTION_HEADERS)

    #preprocess and structure the data
    processed_sections = {
        "filename": os.path.basename(pdf_path),
        "path": os.path.abspath(pdf_path),
        "category": "resume",
        "summary": preprocess(sections.get("summary", "")),
        "experience": preprocess(sections.get("experience", "")),
        "skills": preprocess(sections.get("skills", "")),
        "education": preprocess(sections.get("education", "")),
        "interest": preprocess(sections.get("interests", "")),
        "projects": preprocess(sections.get("projects", "")),
        "certifications": preprocess(sections.get("certifications", "")),
        "publications": preprocess(sections.get("publications", ""))
    }

    ordered_sections = [
        processed_sections["summary"],
        processed_sections["experience"],
        processed_sections["skills"],
        processed_sections["education"],
        processed_sections["interest"],
        processed_sections["projects"],
        processed_sections["certifications"],
        processed_sections["publications"],
    ]

    return ordered_sections
    


#generate embeddings from resume sections
def generate_embeddings(sections_dict, model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"):
    print("Generating embeddings for resume sections...")
    
    encoder = EncoderWithChunks(model_name=model_name, framework="pt", device=device)

    section_keys = [
        "summary", "experience", "skills", "education",
        "interest", "projects", "certifications", "publications"
    ]

    embeddings = []
    for key in section_keys:
        text = sections_dict.get(key, "")
        if not text.strip():
            embeddings.append([])
        else:
            encoded = encoder.encode(text)
            serialized = [e.tolist() for e in encoded]
            embeddings.append(serialized)

    del encoder
    torch.cuda.empty_cache()

    return embeddings


#match the resume to job descriptions
def match_to_jobs(embedding_csv_path):
    print("Matching resume to job descriptions...")


def main():
    input_pdf = "/Users/sumukharadhya/Downloads/273_NLP/capstone/nlp-capstone-project/resume/data/data/ENGINEERING/10030015.pdf"

    #extract text from PDF resume into structured CSV
    structured_resume_csv = extract_resume_sections(input_pdf)
    print(structured_resume_csv)

    #convert to dict with proper keys
    section_names = [
        "summary", "experience", "skills", "education",
        "interest", "projects", "certifications", "publications"
    ]
    sections_dict = dict(zip(section_names, structured_resume_csv))

    #generate embeddings
    embeddings = generate_embeddings(sections_dict)

    for i, section in enumerate(section_names):
        print("\nEmbeddings for section ",section,":")
        print(embeddings[i][:1]) 




if __name__ == "__main__":
    main()
