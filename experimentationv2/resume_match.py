import json
import math
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
        if not true_indices:  # still empty after slicing
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
    return filename.replace(".txt", "")

# Load ground truth
gt_df = pd.read_csv("golden_set.csv")

def get_ground_truth(row):
    return [
        row["top_1_jd_index"],
        row["top_2_jd_index"],
        row["top_3_jd_index"]
    ]

ground_truth = {
    row["resume_filename"] : get_ground_truth(row)
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
    # TODO
    # during inference: For a new resume that comes in, find the resume_category uding jd_category.py (LLM call)
    safe_model_name = model.replace("/", "_").replace("-", "_")
    embedding_model_name = model.replace("/", "_")
    EMB_DIM = cls_dimensions[model]

    jd_data = pd.read_csv(f"../jd_embeddings/jd_{embedding_model_name}.csv")

    jd_original = pd.read_csv("jd_data.csv")
    jd_data = pd.merge(jd_data, jd_original[["position_title", "category"]], on="position_title", how="left")

    # Flatten JD section embeddings
    all_jd_cols = {col for jd_cols in section_dict.values() for col in (jd_cols if isinstance(jd_cols, list) else [jd_cols])}
    for jd_col in all_jd_cols:
        jd_data[jd_col] = jd_data[jd_col].apply(lambda x: decode_and_pool_embedding(x, EMB_DIM))
    jd_vectors_dict = {col: np.stack(jd_data[col].values) for col in all_jd_cols}

    df = pd.read_csv(f"df_resume_{safe_model_name}.csv")
    for res_col in section_dict.keys():
        df[res_col] = df[res_col].apply(lambda x: decode_and_pool_embedding(x, EMB_DIM))
    resume_vectors_dict = {col: np.stack(df[col].values) for col in section_dict.keys()}

    # Prepare category lists
    resume_categories_list = df["category"].fillna("Unknown").str.strip().str.upper().values
    jd_categories_list = jd_data["category"].fillna("Unknown").str.strip().str.upper().values

    resume_categories = {
        normalize_filename(row["filename"]): row.get("category", "Unknown")
        for _, row in df.iterrows()
    }
    
    print("Computing similarities with category filtering...")

    similarities = np.full((len(df), len(jd_data)), -np.inf)

    for resume_idx, resume_category in enumerate(resume_categories_list):
        for jd_idx, jd_category in enumerate(jd_categories_list):
            if resume_category != jd_category:
                continue  # Skip if categories don't match

            # Otherwise, compute similarity between resume_idx and jd_idx
            sim_score = 0
            for res_col, jd_cols in section_dict.items():
                weight = section_weights.get(res_col, 1.0)
                jd_cols = jd_cols if isinstance(jd_cols, list) else [jd_cols]

                sim_sum = 0
                for jd_col in jd_cols:
                    sim_matrix = cosine_similarity(
                        resume_vectors_dict[res_col][resume_idx].reshape(1, -1),
                        jd_vectors_dict[jd_col][jd_idx].reshape(1, -1)
                    )[0, 0]
                    sim_sum += sim_matrix
                sim_avg = sim_sum / len(jd_cols)
                sim_score += weight * sim_avg

            similarities[resume_idx, jd_idx] = sim_score

    top_k = 10
    top_matches = np.argsort(-similarities, axis=1)[:, :top_k]
    predictions = {
        normalize_filename(df.iloc[i]["filename"]): top_matches[i].tolist()
        for i in range(len(df))
    }

    print(f"\nEvaluation Metrics for {model}:")
    metrics = evaluate_predictions(predictions, ground_truth, resume_categories, k=top_k, ground_truth_top_k=3)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    output_rows = []

    print(f"Writing results:")
    for resume_idx, job_indices in enumerate(top_matches):
        filename = normalize_filename(df.iloc[resume_idx]["filename"])
        if filename not in ground_truth:
            continue
        resume_category = df.iloc[resume_idx].get("category", "Unknown")
        if resume_categories in EXCLUDED_CATEGORIES:
            continue
        gold_indices = ground_truth[filename]
        print(gold_indices)
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

