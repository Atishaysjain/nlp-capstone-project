import json, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier          
from xgboost   import XGBClassifier                          
import seaborn as sns
import os

models = [
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
][::-1]  

EMB_COLS = [
    "experience-embeddings", "skills-embeddings",
    "education-embeddings"
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


def is_missing(x):
    """True when x is empty / None / NaN / zero-length array / empty string."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    if isinstance(x, (list, np.ndarray)) and len(x) == 0:
        return True
    return False


def pool(cell, zero_vec: np.ndarray) -> np.ndarray:
    """
    Convert a JSON-encoded or native object to a single resume-section vector.

    • If *cell* is a JSON string  → decode first.
    • If it’s a list of floats    → return it as 1-D vector.
    • If it’s a list[ list[float] ] (chunks) → stack & mean-pool.
    • If missing / empty          → return zero_vec.
    """
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except json.JSONDecodeError:
            pass                              # leave as-is if it isn’t valid JSON

    if cell is None or (isinstance(cell, float) and math.isnan(cell)) or cell == []:
        return zero_vec

    if isinstance(cell, np.ndarray):
        return cell.astype(np.float32) if cell.ndim == 1 else cell.mean(0).astype(np.float32)

    if isinstance(cell, list):
        # Single chunk → list of floats
        if cell and not isinstance(cell[0], (list, tuple)):
            return np.asarray(cell, dtype=np.float32)

        # Multiple chunks → list of list-of-floats
        mat = np.vstack([np.asarray(chunk, dtype=np.float32) for chunk in cell])
        return mat.mean(axis=0).astype(np.float32)

    raise TypeError(f"Unsupported cell type: {type(cell)}")


def sanitize_filename(name):
    return name.replace("/", "_").replace("\\", "_")


if __name__ == "__main__":

    classification_result_dir = os.path.join(os.getcwd(), 'classification_results')

    for model in models:
        resume_emb_input_dir = os.path.join(os.getcwd(), 'resume_embeddings')
        var_name = "df_resume_" + model.replace("/", "_").replace("-", "_")
        df = pd.read_csv(os.path.join(resume_emb_input_dir, var_name + '.csv'))

        EMB_DIM = cls_dimensions[model]

        zero = np.zeros(EMB_DIM, dtype=np.float32)      # reusable filler vector

        for col in EMB_COLS:
            df[col] = df[col].apply(lambda cell: pool(cell, zero))

        df["has_missing"] = df[EMB_COLS].apply(lambda row: any(is_missing(cell) or np.allclose(cell, zero) for cell in row), axis=1)
        df = df[~df["has_missing"]].drop(columns=["has_missing"])

        X = np.stack(df[EMB_COLS].apply(lambda row: np.concatenate(row.values), axis=1).to_numpy())

        le = LabelEncoder()
        y = le.fit_transform(df["category"])

        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")
        num_classes = len(le.classes_) # number of unique labels
        print(f"Number of classes: {num_classes}")

        print("model:", model)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
        models = {
            "LightGBM": LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                class_weight="balanced",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=256,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                tree_method="gpu_hist",     
                predictor="gpu_predictor",
                gpu_id=0,    
                n_estimators=400,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42
            ),
            "LogReg-multinomial": LogisticRegression(
                max_iter=10_000, n_jobs=-1, penalty="l2",
                multi_class="multinomial", class_weight="balanced"
            )
        }

        safe_model = sanitize_filename(model)
        results = []
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="macro")
            results.append((name, acc, f1))

            print(f"─ {name} ───────────────────────────────────────────────────")
            print(f"Accuracy  : {acc:.3f}")
            print(f"Macro-F1  : {f1:.3f}\n")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df = df.drop(columns=['support'])
            df = df.loc[[label for label in le.classes_] + [col for col in df.index if col not in le.classes_]]
            plt.figure(figsize=(10, len(df) * 0.6))
            sns.heatmap(df, annot=True, cmap="Blues", fmt=".2f", linewidths=.5)
            plt.title("Classification Report Heatmap for classifier " + name + " on " + model)
            plt.yticks(rotation=0)
            plt.tight_layout()
            safe_name = sanitize_filename(name)
            save_dir = os.path.join(classification_result_dir, 'classification_reports')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'classification_report_heatmap_{safe_model}_{safe_name}.png'))
            plt.close()

        print("\nSummary (sorted by Macro-F1):")
        for name, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True):
            print(f"{name:<18}  acc={acc:.3f}   macro-F1={f1:.3f}")
        
        results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Macro_F1'])
        results_df['Embedding_Model'] = model        
        results_file = os.path.join(classification_result_dir, 'model_comparison_results.csv')
        try:
            existing_results = pd.read_csv(results_file)
            combined_results = pd.concat([existing_results, results_df], ignore_index=True)
            combined_results.to_csv(results_file, index=False)
        except FileNotFoundError:
            results_df.to_csv(results_file, index=False)
        
        print(f"Results saved to {results_file}")


       