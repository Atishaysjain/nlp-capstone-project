import json, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

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

# 2️⃣  — utility: mean-pool a list of arrays (or return zeros if empty)
def pool(cell, zero_vec: np.ndarray) -> np.ndarray:
    """
    Convert a JSON-encoded or native object to a single resume-section vector.

    • If *cell* is a JSON string  → decode first.
    • If it’s a list of floats    → return it as 1-D vector.
    • If it’s a list[ list[float] ] (chunks) → stack & mean-pool.
    • If missing / empty          → return zero_vec.
    """
    # ─── 1. Decode JSON strings ────────────────────────────────────────────────────
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except json.JSONDecodeError:
            pass                              # leave as-is if it isn’t valid JSON

    # ─── 2. Handle empty / NaN / None ─────────────────────────────────────────────
    if cell is None or (isinstance(cell, float) and math.isnan(cell)) or cell == []:
        return zero_vec

    # ─── 3. Already a NumPy vector? ───────────────────────────────────────────────
    if isinstance(cell, np.ndarray):
        return cell.astype(np.float32) if cell.ndim == 1 else cell.mean(0).astype(np.float32)

    # ─── 4. Plain Python list(s) ──────────────────────────────────────────────────
    if isinstance(cell, list):
        # Single chunk → list of floats
        if cell and not isinstance(cell[0], (list, tuple)):
            return np.asarray(cell, dtype=np.float32)

        # Multiple chunks → list of list-of-floats
        mat = np.vstack([np.asarray(chunk, dtype=np.float32) for chunk in cell])
        return mat.mean(axis=0).astype(np.float32)

    # ─── 5. If we reach here we got an unexpected type ────────────────────────────
    raise TypeError(f"Unsupported cell type: {type(cell)}")


if __name__ == "__main__":

    for model in models:
        var_name = "df_resume_" + model.replace("/", "_").replace("-", "_")
        df = pd.read_csv(var_name + '.csv')

        EMB_DIM = cls_dimensions[model]

        zero = np.zeros(EMB_DIM, dtype=np.float32)      # reusable filler vector

        # 3️⃣  — apply to each embedding column
        for col in EMB_COLS:
            df[col] = df[col].apply(lambda cell: pool(cell, zero))

        # 4️⃣  — concatenate the eight section vectors ➜ one resume vector
        X = np.stack(df[EMB_COLS].apply(lambda row: np.concatenate(row.values), axis=1).to_numpy())

        # 5️⃣  — encode string labels
        le = LabelEncoder()
        y = le.fit_transform(df["category"])

        # 6️⃣  — train / evaluate

        print("model:", model)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        models = {
            "LogReg-multinomial": LogisticRegression(
                max_iter=10_000, n_jobs=-1, penalty="l2",
                multi_class="multinomial", class_weight="balanced"
            ),
            "Linear SVM": LinearSVC(C=1.0, class_weight="balanced"),
            "SGD-log": SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced"),
            "MLP(512,256)": MLPClassifier(
                hidden_layer_sizes=(512,256), alpha=1e-4,
                early_stopping=True, random_state=42
            ),
        }

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
            # Uncomment for full per-class details
            # print(classification_report(y_test, y_pred, target_names=le.classes_))

        # ─────────────────────────────────── summary table (nicely sorted) ────────────────
        print("\nSummary (sorted by Macro-F1):")
        for name, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True):
            print(f"{name:<18}  acc={acc:.3f}   macro-F1={f1:.3f}")

       