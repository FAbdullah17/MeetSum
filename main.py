# main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid

# Import your inference engine (located in app/)
from app.inference_engine import summarize_long_text, read_transcript_file

app = FastAPI(
    title="MeetSum",
    description="Summarizes meeting transcripts using a fine-tuned BART model.",
    version="1.0"
)

# === CORS Middleware (optional) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Serve static files (HTML/CSS) ===
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Uploads Directory ===
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Serve Frontend UI ===
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return HTMLResponse("<h1>🔧 Frontend not found. Please make sure index.html is in /static.</h1>", status_code=500)

# === API: Summarize Text ===
@app.post("/summarize/text")
async def summarize_text(text: str = Form(...)):
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "Empty text input."})

    try:
        summary = summarize_long_text(text)
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === API: Summarize Uploaded File ===
@app.post("/summarize/file")
async def summarize_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".pdf", ".docx"]:
        return JSONResponse(status_code=400, content={"error": "Unsupported file format. Use .txt, .pdf, or .docx."})

    try:
        # Save uploaded file temporarily
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract content and summarize
        transcript = read_transcript_file(file_path)
        if not transcript.strip():
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty or unreadable."})

        summary = summarize_long_text(transcript)

        # Clean up uploaded file
        os.remove(file_path)

        return {"summary": summary}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
