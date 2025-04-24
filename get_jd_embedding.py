import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import EncoderWithChunks
import json

if __name__ == "__main__":

    jd_df = pd.read_csv('training_data.csv')

    models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "albert-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ][::-1] 

    for model in models:
        encoder = EncoderWithChunks(model_name=model, framework="pt", device="cuda")
        # generate embeddings for all the job descriptions
        embedding_list = []
        for job_desc in jd_df["job_description"]:
            encoded = encoder.encode(job_desc)
            serialized = [e.tolist() for e in encoded]
            embedding_list.append(json.dumps(serialized))
            # embedding_list.append(encoder.encode(job_desc))
            
        # converting the list of embeddings to a numpy array
        jd_df[f"{model}-embeddings"] = embedding_list
        print(f"Model {model} finished encoding job descriptions.")

        # save the embeddings to a CSV file
        jd_df.to_csv(f"training_data_embeddings.csv", index=False)

        del encoder