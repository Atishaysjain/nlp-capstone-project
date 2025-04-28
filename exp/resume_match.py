import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

models = [
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
][::-1]  # Reverse the order of models

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

def evaluate_predictions(predictions: dict, ground_truth: dict, resume_categories: dict, k: int = 25, ground_truth_top_k: int = 5):
    top1_hits = 0
    mrr = []
    evaluated_count = 0
    top5_coverage = []
    top5_hits = []

    for filename, pred_indices in predictions.items():
        category = resume_categories.get(filename, "Unknown").upper()
        if category in EXCLUDED_CATEGORIES:
            continue

        true_indices = ground_truth.get(filename, [])
        if not true_indices:
            continue

        true_indices = true_indices[:ground_truth_top_k]

        # Top-1 Accuracy (is top-1 prediction in top-5 GT)
        if pred_indices[0] in true_indices:
            top1_hits += 1

        top5_pred = set(pred_indices[:k])
        top5_truth = set(true_indices[:ground_truth_top_k])
        top5_match_count = len(top5_pred & top5_truth)
        top5_hits.append(top5_match_count / 5)

        # Top-5 Recall: how many of the top-5 GT are found in top-25 predictions
        top5_covered = len(set(true_indices[:5]) & set(pred_indices[:25]))
        top5_coverage.append(top5_covered / 5)

        # MRR: rank of first correct match in top 25
        reciprocal_rank = 0
        for rank, pred in enumerate(pred_indices[:k], start=1):
            if pred in true_indices:
                reciprocal_rank = 1 / rank
                break
        mrr.append(reciprocal_rank)

        evaluated_count += 1

    return {
        "Top-1 Accuracy": top1_hits / evaluated_count if evaluated_count else 0,
        "Top-5 GT Covered in Top-25 Preds": sum(top5_coverage) / evaluated_count if evaluated_count else 0,
        "MRR": sum(mrr) / evaluated_count if evaluated_count else 0,
        "Evaluated Samples": evaluated_count
    }

def normalize_filename(filename):
    return filename.replace(".pdf", "").replace(".txt", "")

# Load ground truth
gt_df = pd.read_csv("resume_matches_output.csv")

def get_ground_truth(row):
    return [
        row["top match JD 1 index"],
        row["top match JD 2 index"],
        row["top match JD 3 index"],
        row["top match JD 4 index"],
        row["top match JD 5 index"],
    ]

ground_truth = {
    normalize_filename(row["filename"]): get_ground_truth(row)
    for _, row in gt_df.iterrows()
}

# Mapping resume section → JD section
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

# Weights per resume section
section_weights = {
    "skills-embeddings": 0.45,
    "education-embeddings": 0.1,
    "experience-embeddings": 0.35,
    "summary-embeddings": 0.1
}

for model in models:
    safe_model_name = model.replace("/", "_").replace("-", "_")
    embedding_model_name = model.replace("/", "_")
    EMB_DIM = cls_dimensions[model]

    jd_data = pd.read_csv(f"jd_embeddings/jd_{embedding_model_name}.csv")
    
    # Flatten JD section embeddings
    all_jd_cols = {col for jd_cols in section_dict.values() for col in (jd_cols if isinstance(jd_cols, list) else [jd_cols])}
    for jd_col in all_jd_cols:
        jd_data[jd_col] = jd_data[jd_col].apply(lambda x: decode_and_pool_embedding(x, EMB_DIM))
    jd_vectors_dict = {col: np.stack(jd_data[col].values) for col in all_jd_cols}

    df = pd.read_csv(f"df_resume_{safe_model_name}.csv")
    for res_col in section_dict.keys():
        df[res_col] = df[res_col].apply(lambda x: decode_and_pool_embedding(x, EMB_DIM))
    resume_vectors_dict = {col: np.stack(df[col].values) for col in section_dict.keys()}

    similarities = np.zeros((len(df), len(jd_data)))
    for res_col, jd_cols in section_dict.items():
        weight = section_weights.get(res_col, 1.0)
        jd_cols = jd_cols if isinstance(jd_cols, list) else [jd_cols]

        print(f"Computing similarity for: {res_col} ↔ {jd_cols} (weight: {weight})")
        sim_sum = np.zeros_like(similarities)
        for jd_col in jd_cols:
            sim_matrix = cosine_similarity(resume_vectors_dict[res_col], jd_vectors_dict[jd_col])
            sim_sum += sim_matrix
        sim_avg = sim_sum / len(jd_cols)  # average if multiple mappings
        similarities += weight * sim_avg

    top_k = 25
    top_matches = np.argsort(-similarities, axis=1)[:, :top_k]
    predictions = {
        normalize_filename(df.iloc[i]["filename"]): top_matches[i].tolist()
        for i in range(len(df))
    }

    resume_categories = {
        normalize_filename(row["filename"]): row.get("category", "Unknown")
        for _, row in df.iterrows()
    }

    metrics = evaluate_predictions(predictions, ground_truth, resume_categories, k=top_k, ground_truth_top_k=5)

    print(f"\nEvaluation Metrics for {model}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    output_rows = []
    for resume_idx, job_indices in enumerate(top_matches):
        filename = normalize_filename(df.iloc[resume_idx]["filename"])
        if filename not in ground_truth:
            continue
        resume_category = df.iloc[resume_idx].get("category", "Unknown")
        gold_indices = ground_truth[filename]
        for rank, job_idx in enumerate(job_indices, start=1):
            output_rows.append({
                "resume_filename": filename,
                "resume_category": resume_category,
                "gold_jd_indices": gold_indices,
                "predicted_rank": rank,
                "predicted_jd_index": job_idx,
                "predicted_jd_title": jd_data.iloc[job_idx].get("position_title", "N/A"),
                "predicted_jd_company": jd_data.iloc[job_idx].get("company_name", "N/A"),
                "predicted_jd_description": jd_data.iloc[job_idx].get("job_description", "N/A"),
                "similarity_score": similarities[resume_idx][job_idx]
            })

    results_df = pd.DataFrame(output_rows)
    results_df.to_csv(f"predictions_{safe_model_name}.csv", index=False)
    print(f"Saved results to predictions_{safe_model_name}.csv")
