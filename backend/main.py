from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- 1. Import this
from pypdf import PdfReader
import io
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="HireAI Backend", version="1.0")

# --- 2. Add CORS Middleware here ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows our Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Securely retrieve the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables or .env file!")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

@app.get("/")
def read_root():
    return {"message": "Welcome to HireAI Backend! The server is running securely."}

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are accepted.")
    
    try:
        # 1. Extract text using pypdf
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        reader = PdfReader(pdf_file)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
                
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. It might be scanned or empty.")
            
        # 2. Construct a professional prompt for Gemini
        prompt = f"""
        You are an expert technical recruiter and resume reviewer. 
        Analyze the following resume text and provide:
        1. A score out of 100.
        2. Top 3 strengths.
        3. Top 3 areas for improvement.
        4. Missing key sections or keywords.

        Resume Text:
        {extracted_text}
        """

        # 3. Call Gemini model
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )

        return {
            "filename": file.filename,
            "ai_analysis": response.text,
            "full_text": extracted_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the resume: {str(e)}")