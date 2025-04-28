import numpy as np
import pandas as pd
from preprocessor import EncoderWithChunks
import json
from tqdm import tqdm 
import torch
from collections.abc import Iterable 
import os


# Function to encode text and return JSON string of chunk embeddings
def encode_text_list_to_json(encoder, text_list, desc="Encoding"):
    """
    Encodes a list of texts, returning a list where each element is a
    JSON string representing the list of chunk embeddings for the corresponding text.
    """
    all_serialized_embeddings = []

    for text in tqdm(text_list, desc=desc, leave=False):
        # --- Handle potential list/array inputs ---
        if text is None:
            text_str = ""
        # Check if text is an iterable (list, array, etc.) but NOT a string
        elif isinstance(text, Iterable) and not isinstance(text, (str, bytes)):
            text_str = ". ".join(str(item) for item in text if pd.notna(item))
        else:
            # Assume it's a scalar (string, number, etc.) or bytes that can be converted
            text_str = str(text)
        # --- End Handling ---

        # Encode the text, returns a list of np.arrays (one per chunk)
        chunk_embeddings = encoder.encode(text_str) # Pass the processed string

        # Convert each chunk embedding (np.array) to a list
        serialized_chunks = [e.tolist() for e in chunk_embeddings]

        # Convert the list of lists (or list of chunk embeddings) into a JSON string
        json_string_of_chunks = json.dumps(serialized_chunks)

        all_serialized_embeddings.append(json_string_of_chunks)

    return all_serialized_embeddings


if __name__ == "__main__":

    input_jd_file_path = 'jd_data.csv'

    jd_df_original = pd.read_csv(input_jd_file_path)

    print("Parsing model responses...")
    parsed_data = {
        'Compensation_Benefits': [],
        'Core_Responsibilities': [],
        'Educational_Requirements': [],
        'Experience_Level': [],
        'Preferred_Qualifications': [],
        'Required_Skills': []
    }
    json_keys = list(parsed_data.keys())
    json_keys_in_file = [
        'Compensation and Benefits', 'Core Responsibilities',
        'Educational Requirements', 'Experience Level',
        'Preferred Qualifications', 'Required Skills'
    ]

    for i, model_response_str in enumerate(jd_df_original["model_response"]):
        try:
            # Load the JSON string into a dictionary
            model_response_dict = json.loads(model_response_str)
            for key_internal, key_file in zip(json_keys, json_keys_in_file):
                 parsed_data[key_internal].append(model_response_dict.get(key_file, None))
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Invalid JSON/data at index {i}. Appending None for all fields.")
            for key in json_keys:
                parsed_data[key].append(None) 

    models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "albert-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    models.reverse() 

    fields_to_encode = ['job_description'] + json_keys

    print(f"Processing models: {models}")
    print(f"Fields to encode: {fields_to_encode}")

    for model_name in tqdm(models, desc="Processing Models"):
        print(f"\nProcessing model: {model_name}")

        model_df = jd_df_original.copy()

        encoder = EncoderWithChunks(model_name=model_name, framework="pt", device="cuda")

        for field in fields_to_encode:
            print(f"  Encoding field: {field}")

            if field == 'job_description':
                text_list = model_df[field].fillna("").tolist()
            else:
                text_list = parsed_data[field] # Use the parsed data (already handles Nones)

            num_texts = len(text_list)

            list_of_json_strings = encode_text_list_to_json(encoder, text_list, desc=f"  {field}")

            print(f"    Generated {len(list_of_json_strings)} JSON strings for embeddings.")

            if len(list_of_json_strings) != num_texts:
                 print(f"    Warning: Number of generated JSON strings ({len(list_of_json_strings)}) mismatch with text count ({num_texts}) for field '{field}'.")
                 continue


            model_df[f"{field}-embedding"] = list_of_json_strings
            print(f"    Finished encoding field '{field}'.")

        safe_model_name = model_name.replace('/', '_')
        output_dir = os.path.join(os.getcwd(), 'jd_embeddings')
        output_file = os.path.join(output_dir, f"jd_{safe_model_name}.csv")
        print(f"  Saving results for model {model_name} to {output_file}...")
        try:
            model_df.to_csv(output_file, index=False)
            print(f"  Save complete for model {model_name}.")
        except Exception as e:
            print(f"  Error saving file {output_file}: {e}")


        del encoder
        del model_df
        if torch.cuda.is_available():
           torch.cuda.empty_cache()


    print("\nAll models processed.")
