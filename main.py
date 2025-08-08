import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# This is our "brain" file we created in the last step
from rag_processor import generate_answers_from_document

# Load environment variables
print("INFO: Loading environment variables...")
load_dotenv()

# Configure Google Generative AI
import google.generativeai as genai
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable not found.")
else:
    print("INFO: Google API Key loaded successfully.")
    genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(title="HackRx 6.0 RAG API")

# Request schema
class HackathonRequest(BaseModel):
    documents: str
    questions: list[str]

# Response schema
class HackathonResponse(BaseModel):
    answers: list[str]

@app.get("/", summary="Check if the API is running")
def read_root():
    return {"status": "ok", "message": "Welcome to the HackRx 6.0 API!"}

@app.post("/hackrx/run", response_model=HackathonResponse, summary="Process a document and questions")
async def run_logic(request: HackathonRequest):
    print("INFO: Received request on /hackrx/run")
    if not api_key:
        raise HTTPException(status_code=500, detail="Server is not configured with a Google API Key.")

    try:
        print("INFO: Handing off request to the RAG processor...")
        answers = generate_answers_from_document(
            pdf_url=request.documents,
            questions=request.questions
        )
        print("INFO: RAG processor returned answers. Sending response.")
        return HackathonResponse(answers=answers)

    except Exception as e:
        print(f"CRITICAL ERROR during RAG processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# 🔥 This is needed for Render to bind to $PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
