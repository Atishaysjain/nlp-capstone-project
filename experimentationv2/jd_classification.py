from google import genai
import time
import json
import pandas as pd

# Initialize Gemini client
client = genai.Client(api_key="API_KEY")
generation_config={"response_mime_type": "application/json"}

# Helper functions
def call_llm(prompt, max_retries=3, delay=60, model="gemini-2.0-flash", temperature=0.1):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=generation_config)
            return response
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    return None

def classify_category(core_responsibilities_text):
    prompt = f"""You are a classification expert. Classify the following job description into a category:
    
    Core Responsibilities: {core_responsibilities_text}
    
    Categories:
    - ACCOUNTANT
    - AVIATION
    - BANKING
    - CHEF
    - CONSTRUCTION
    - DESIGNER
    - ENGINEERING
    - FINANCE
    - HEALTHCARE
    - HR
    - INFORMATION-TECHNOLOGY
    - TEACHER
    - ADVOCATE
    - AGRICULTURE
    - APPAREL
    - ARTS
    - AUTOMOBILE
    - BPO
    - BUSINESS-DEVELOPMENT
    - CONSULTANT
    - DIGITAL-MEDIA
    - PUBLIC-RELATIONS
    - SALES
    - FITNESS
    
    Output the category name in JSON format. Adhere to the format strictly.
    Output Format to follow:
    {{ "category" : "SALES" }}
    """

    response = call_llm(prompt)
    if response:
        try:
            print("🔍 Raw LLM Response:", response.text.strip())
            parsed = json.loads(response.text.strip())
            return parsed.get("category", "Unknown").strip().upper()
        except Exception as e:
            print(f"⚠️ Failed to parse JSON: {e}")
            return "Unknown"
    else:
        return "Unknown"


jd_path = 'training_data.csv'
jd_df = pd.read_csv(jd_path)

def extract_and_classify(row):
    try:
        model_response = json.loads(row["model_response"])
        core_resp = model_response.get("Core Responsibilities", "")
        return classify_category(core_resp)
    except Exception as e:
        print(f"Failed to parse or classify row: {e}")
        return "Unknown"

print("Classifying job descriptions into categories...")
jd_df["category"] = jd_df.apply(extract_and_classify, axis=1)

jd_df.to_csv(jd_path, index=False)
print(f"Saved updated JD file with categories to: {jd_path}")