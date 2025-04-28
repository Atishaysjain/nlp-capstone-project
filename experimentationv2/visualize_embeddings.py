import json, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os
from sklearn.manifold import TSNE 


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


def plot_tsne(X, y_labels, model_name, save_dir):
    """
    Performs t-SNE dimensionality reduction and plots the results.
    """
    print(f"Performing t-SNE for {model_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] - 1), n_iter=300, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_df['Category'] = y_labels # Use original labels for plotting

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x="TSNE1", y="TSNE2",
        hue="Category",
        palette=sns.color_palette("hsv", len(np.unique(y_labels))),
        data=tsne_df,
        legend="full",
        alpha=0.7
    )
    plt.title(f't-SNE visualization of Resume Embeddings ({model_name})')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    safe_model_name = sanitize_filename(model_name)
    tsne_save_path = os.path.join(save_dir, f'tsne_visualization_{safe_model_name}.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(tsne_save_path)
    plt.close()
    print(f"t-SNE plot saved to {tsne_save_path}")


if __name__ == "__main__":

    tsne_visualization_dir = os.path.join(os.getcwd(), 'tsne_visualizations')

    for model in models:
        var_name = "df_resume_" + model.replace("/", "_").replace("-", "_")
        input_emb_dir = os.path.join(os.getcwd(), 'resume_embeddings')
        input_emb_file = os.path.join(input_emb_dir, var_name + '.csv')
        df = pd.read_csv(input_emb_file)

        EMB_DIM = cls_dimensions[model]

        zero = np.zeros(EMB_DIM, dtype=np.float32)     

        for col in EMB_COLS:
            df[col] = df[col].apply(lambda cell: pool(cell, zero))

        df["has_missing"] = df[EMB_COLS].apply(lambda row: any(is_missing(cell) or np.allclose(cell, zero) for cell in row), axis=1)
        df = df[~df["has_missing"]].drop(columns=["has_missing"])

        # Ensure we have enough data points after filtering
        if df.shape[0] < 2:
            print(f"Skipping model {model} due to insufficient data after filtering ({df.shape[0]} samples).")
            continue

        X = np.stack(df[EMB_COLS].apply(lambda row: np.concatenate(row.values), axis=1).to_numpy())

        original_labels = df["category"].values # original labels for plotting
        le = LabelEncoder()
        y = le.fit_transform(original_labels)

        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")
        num_classes = len(le.classes_) # number of unique labels
        print(f"Number of classes: {num_classes}")

        print("model:", model)

        # --- Perform and Plot t-SNE ---
        # Check if we have enough samples for the default perplexity
        if X.shape[0] > 1: # t-SNE requires more than 1 sample
             plot_tsne(X, original_labels, model, tsne_visualization_dir)
        else:
             print(f"Skipping t-SNE for model {model} due to insufficient data points ({X.shape[0]}).")
        # -----------------------------