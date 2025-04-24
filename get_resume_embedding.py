import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import EncoderWithChunks
import torch
import json

if __name__ == "__main__":

    df_resume = pd.read_csv('./resume/data/structured_resumes.csv')

    models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "albert-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ][::-1]  # Reverse the order of models

    def model_to_varname(model_name):
        return "df_resume_" + model_name.replace("/", "_").replace("-", "_")

    print(models)

    for model in models:

        # var_name = model_to_varname(model)
        # globals()[var_name] = df_resume.copy(deep=True)

        df = df_resume.copy(deep=True)
        
        encoder = EncoderWithChunks(model_name=model, framework="pt", device="cuda")
        # generate embeddings for all the job descriptions
        credential_columns = ['summary', 'experience', 'skills',
       'education', 'interest', 'projects', 'certifications', 'publications']
        summary_embedding = []
        experience_embedding = []
        skills_embedding = []
        education_embedding = []
        interest_embedding = []
        projects_embedding = []
        certifications_embedding = []
        publications_embedding = []

        for index, row in df.iterrows():
            if(pd.isna(row['summary'])):
                summary_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['summary'])
                serialized = [e.tolist() for e in encoded]
                summary_embedding.append(json.dumps(serialized))
            if(pd.isna(row['experience'])):
                experience_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['experience'])
                serialized = [e.tolist() for e in encoded]
                experience_embedding.append(json.dumps(serialized))
            if(pd.isna(row['skills'])):
                skills_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['skills'])
                serialized = [e.tolist() for e in encoded]
                skills_embedding.append(json.dumps(serialized))
            if(pd.isna(row['education'])):
                education_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['education'])
                serialized = [e.tolist() for e in encoded]
                education_embedding.append(json.dumps(serialized))
            if(pd.isna(row['interest'])):
                interest_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['interest'])
                serialized = [e.tolist() for e in encoded]
                interest_embedding.append(json.dumps(serialized))
            if(pd.isna(row['projects'])):
                projects_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['projects'])
                serialized = [e.tolist() for e in encoded]
                projects_embedding.append(json.dumps(serialized))
            if(pd.isna(row['certifications'])):
                certifications_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['certifications'])
                serialized = [e.tolist() for e in encoded]
                certifications_embedding.append(json.dumps(serialized))
            if(pd.isna(row['publications'])):
                publications_embedding.append(json.dumps([]))
            else:
                encoded = encoder.encode(row['publications'])
                serialized = [e.tolist() for e in encoded]
                publications_embedding.append(json.dumps(serialized))

        
        df["summary-embeddings"] = summary_embedding
        df["experience-embeddings"] = experience_embedding
        df["skills-embeddings"] = skills_embedding
        df["education-embeddings"] = education_embedding
        df["interest-embeddings"] = interest_embedding
        df["projects-embeddings"] = projects_embedding
        df["certifications-embeddings"] = certifications_embedding
        df["publications-embeddings"] = publications_embedding

        df.to_csv(model_to_varname(model) + ".csv", index=False)

        print(f"Model {model} finished encoding resumes.")

        del encoder
        torch.cuda.empty_cache()