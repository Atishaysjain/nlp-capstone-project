from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from main import extract_resume_sections, generate_embeddings, match_to_jobs

app = FastAPI()

# Allow your frontend origin
origins = [
    "http://localhost:5173",  # Vite default dev server
    "https://resume-matcher-ten.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/get_matches/")
async def get_matches(file: UploadFile = File(...)):
    #save uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #extract and preprocess resume
    section_names = [
        "summary", "experience", "skills", "education",
        "interest", "projects", "certifications", "publications"
    ]
    structured_resume = extract_resume_sections(file_path)
    sections_dict = dict(zip(section_names, structured_resume))

    #generate embeddings
    resume_embeddings = generate_embeddings(sections_dict)

    #TODO: Get resume category
    

    #match to job descriptions
    top_matches = match_to_jobs(
        resume_embeddings,
        resume_filename=file.filename,
        resume_category="uploaded",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    #format output
    output = []
    for match in top_matches:
        output.append({
            "title": match["predicted_jd_title"],
            "company": match["predicted_jd_company"],
            "description": match["predicted_jd_description"],
            "similarity_score": round(match["similarity_score"], 4)
        })

    #delete uploaded file after processing
    os.remove(file_path)

    return JSONResponse(content=output)
