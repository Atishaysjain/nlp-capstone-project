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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")


models = [
    "sentence-transformers/all-MiniLM-L6-v2",
]

EMB_COLS = [
    "summary-embeddings", "experience-embeddings", "skills-embeddings",
    "education-embeddings", "interest-embeddings", "projects-embeddings",
    "certifications-embeddings", "publications-embeddings"
]

cls_dimensions = {
    "bert-base-uncased": 768,
    "bert-large-uncased": 1024,
    "distilbert-base-uncased": 768,
    "roberta-base": 768,
    "albert-base-v2": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384
}

EXCLUDED_CATEGORIES = {
    "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE",
    "BPO", "BUSINESS-DEVELOPMENT", "CONSULTANT", "DIGITAL-MEDIA", "PUBLIC-RELATIONS", "SALES",
    "FITNESS",
}

def decode_and_pool_embedding(cell, dim=768):
    try:
        # Step 1: decode if string
        if isinstance(cell, str):
            cell = json.loads(cell)
        
        # Step 2: if still empty or bad
        if not cell:
            return np.zeros(dim)

        # Step 3: pool multiple chunks
        return np.mean([np.array(c) for c in cell], axis=0)
    
    except Exception as e:
        print(f"Error pooling embedding: {e}")
        return np.zeros(dim)

def evaluate_predictions(predictions: dict, ground_truth: dict, resume_categories: dict, k: int = 5, ground_truth_top_k: int = 5):
    top1_hits = 0
    mrr = []
    evaluated_count = 0
    topk_coverage = []
    topk_hits = []

    clean_ground_truth = {}
    
    for k, v in ground_truth.items():
        cleaned_list = [int(x) for x in v if not (isinstance(x, float) and math.isnan(x))]
        clean_ground_truth[k] = cleaned_list


    for file, pred_indices in predictions.items():
        category = resume_categories.get(file, "Unknown").upper()
        if category in EXCLUDED_CATEGORIES:
            continue

        try:
            filename = int(file)
        except:
            continue

        if filename not in clean_ground_truth:
            continue
        
        evaluated_count += 1
        true_indices = clean_ground_truth.get(filename, [])
        if not true_indices:
            continue

        true_indices = true_indices[:ground_truth_top_k]
        if not true_indices:
            continue

        actual_gt_count = len(true_indices)

        # Top-1 Accuracy: is top-1 prediction in ground truth?
        if pred_indices[0] in true_indices:
            top1_hits += 1

        topk_pred = set(pred_indices[:k])
        topk_truth = set(true_indices)

        topk_match_count = len(topk_pred & topk_truth)
        topk_hits.append(topk_match_count / actual_gt_count)  # divide by actual number of GT

        # Top-k Recall: how many ground truth items were found in top-k predictions
        topk_covered = len(topk_pred & topk_truth)
        topk_coverage.append(topk_covered / actual_gt_count)

        # MRR: rank of first correct match
        reciprocal_rank = 0
        for rank, pred in enumerate(pred_indices[:k], start=1):
            if pred in true_indices:
                reciprocal_rank = 1 / rank
                break
        mrr.append(reciprocal_rank)

    return {
        "Top-1 Accuracy": top1_hits / evaluated_count if evaluated_count else 0,
        "Top-3 GT Covered in Top-10 Preds": sum(topk_coverage) / evaluated_count if evaluated_count else 0,
        "MRR": sum(mrr) / evaluated_count if evaluated_count else 0,
        "Evaluated Samples": evaluated_count
    }


def normalize_filename(filename):
    return filename.replace(".pdf", "").replace(".txt", "")

gt_df = pd.read_csv("resume_matches_output.csv")

def get_ground_truth(row):
    return [
        row["top match JD 1 index"],
        row["top match JD 2 index"],
        row["top match JD 3 index"]
    ]

ground_truth = {
    normalize_filename(row["filename"]): get_ground_truth(row)
    for _, row in gt_df.iterrows()
}

#mapping resume section → JD section
section_dict = {
    "skills-embeddings": [
        "Required_Skills-embedding"
    ],
    "education-embeddings": [
        "Educational_Requirements-embedding"
    ],
    "experience-embeddings": [
        "Experience_Level-embedding",
        "Core_Responsibilities-embedding"
    ],
    "summary-embeddings": [
        "Core_Responsibilities-embedding",
        "Experience_Level-embedding",
    ]
}

#weights per resume section
section_weights = {
    "skills-embeddings": 0.45,
    "education-embeddings": 0.1,
    "experience-embeddings": 0.35,
    "summary-embeddings": 0.1
}


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
def generate_embeddings(sections_dict, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
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
def match_to_jobs(resume_embeddings, resume_filename="unknown_resume.pdf", resume_category="unknown", model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    print("Matching resume to job descriptions...")

    embedding_model_name = model_name.replace("/", "_")
    EMB_DIM = cls_dimensions[model_name]

    #load JD data and decode their embeddings
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jd_path = os.path.join(base_dir, "jd_embeddings", f"jd_{embedding_model_name}.csv")
    jd_data = pd.read_csv(jd_path)
    jd_original = pd.read_csv("../exp/training_data.csv")
    jd_data = pd.merge(jd_data, jd_original[["position_title", "category"]], on="position_title", how="left")
    
    #decode JD section embeddings
    all_jd_cols = {col for jd_cols in section_dict.values() for col in (jd_cols if isinstance(jd_cols, list) else [jd_cols])}
    for jd_col in all_jd_cols:
        jd_data[jd_col] = jd_data[jd_col].apply(lambda x: decode_and_pool_embedding(x, EMB_DIM))
    jd_vectors_dict = {col: np.stack(jd_data[col].values) for col in all_jd_cols}

    #get JD categories
    jd_categories_list = jd_data["category"].fillna("Unknown").str.strip().str.upper().values

    #prepare resume embedding dict
    section_keys = list(section_dict.keys())
    resume_vectors_dict = {}
    for i, section in enumerate(section_keys):
        pooled_vector = decode_and_pool_embedding(resume_embeddings[i], EMB_DIM)
        resume_vectors_dict[section] = pooled_vector.reshape(1, -1)

    #compute weighted similarity only with JDs of matching category
    num_jds = len(jd_data)
    similarities = np.zeros(num_jds)

    for res_col, jd_cols in section_dict.items():
        weight = section_weights.get(res_col, 1.0)
        jd_cols = jd_cols if isinstance(jd_cols, list) else [jd_cols]

        for jd_col in jd_cols:
            res_vector = resume_vectors_dict[res_col]
            jd_vector = jd_vectors_dict[jd_col]

            #compute similarity
            sim_matrix = cosine_similarity(res_vector, jd_vector)[0]  #shape: (num_jds,)
            for idx in range(num_jds):
                if resume_category.strip().upper() == jd_categories_list[idx]:
                    similarities[idx] += weight * sim_matrix[idx]

    #get top 5 matches
    top_indices = np.argsort(-similarities)[:5]
    output_rows = []

    for rank, job_idx in enumerate(top_indices, start=1):
        output_rows.append({
            "resume_filename": resume_filename,
            "resume_category": resume_category,
            "gold_jd_indices": [],
            "predicted_rank": rank,
            "predicted_jd_index": job_idx,
            "predicted_jd_title": jd_data.iloc[job_idx].get("position_title", "N/A"),
            "predicted_jd_company": jd_data.iloc[job_idx].get("company_name", "N/A"),
            "predicted_jd_description": jd_data.iloc[job_idx].get("job_description", "N/A"),
            "similarity_score": similarities[job_idx]
        })

    return output_rows

def write_matches_to_csv(top_matches, output_path="top_resume_matches.csv"):
    with open(output_path, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["title", "company", "description", "similarity_score"])
        writer.writeheader()
        for match in top_matches:
            writer.writerow({
                "title": match["predicted_jd_title"],
                "company": match["predicted_jd_company"],
                "description": match["predicted_jd_description"],
                "similarity_score": round(match["similarity_score"], 4)
            })
    print("Top matches written to: ", output_path)



def main():
    input_pdf = "/Users/sumukharadhya/Downloads/273_NLP/capstone/nlp-capstone-project/resume/data/data/ENGINEERING/10030015.pdf"
    resume_filename = os.path.basename(input_pdf)
    resume_category = "ENGINEERING"

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
    
    top_matches = match_to_jobs(
        embeddings,
        resume_filename=resume_filename,
        resume_category=resume_category,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    print("\nTop 5 Matching Job Descriptions:\n")
    for match in top_matches:
        print(f"Rank {match['predicted_rank']}: {match['predicted_jd_title']} at {match['predicted_jd_company']}")
        print(f"    ↳ Similarity: {match['similarity_score']:.4f}")
        print(f"    ↳ Description: {match['predicted_jd_description'][:150]}...\n")
    
    write_matches_to_csv(top_matches)




if __name__ == "__main__":
    main()
