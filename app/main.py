from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
# Load .env automatically when present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; environment variables can be set externally
    pass
from google import genai
from .function import nlp_pipeline
import pandas as pd
from io import BytesIO

app = FastAPI(title="FastAPI Audio Transcription & Summarization")

MODEL_NAME = "gemini-2.5-flash"


# --- Lazy Gemini client initialization ---
def get_genai_client():
    """Return a genai.Client, constructing it on first use.

    This avoids requiring credentials at module import time which
    prevents uvicorn from failing to import the app when credentials
    are not configured in the environment during development.
    """
    # Prefer an explicit API key from environment for safety
    api_key = (
        os.getenv("GENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "GENAI API key not found. Set the environment variable GENAI_API_KEY before calling the endpoint."
        )
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Error initializing Gemini client: {e}") from e

# --- Prompts for transcription and summarization ---
TRANSCRIPTION_PROMPT = (
    "Transcribe the provided audio file accurately with timestamps. "
    "For each spoken segment, include the start time in [MM:SS] format at the beginning of the line. "
    "Example format:\n"
    "[00:00] Speaker: Hello, welcome to the meeting\n"
    "[00:05] Speaker: Today we'll discuss...\n"
    "Ensure accurate timing and capture all spoken words, including speaker changes if identifiable."
)

SUMMARY_PROMPT = (
    """Extract the following information from the provided conversation text: 1.**Room Type**: Identify and extract the type of room being discussed (e.g., single, double, shared).2.**Cost**: Determine the cost associated with the room or accommodation mentioned.3.**Location**: Extract the geographical location of the accommodation.4.**Status of Inhabitant**: Specify whether the tenant is a student or a working professional, and if applicable, include the name of the company or institution they are associated with.5.**Required Amenities**: List any specific amenities that the tenant requires or is looking for in the accommodation (e.g., Wi-Fi, laundry, kitchen access).6.**Alternative Suggestions**: Note any other hostel or accommodation options suggested to the tenant by the landlord during the conversation.Ensure that the extraction is accurate and concise, providing clear labels for each piece of information collected."""
)


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """Upload audio file, transcribe it, and extract structured insights"""

    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Initialize Gemini client
        client = get_genai_client()
        audio_file = client.files.upload(file=temp_path)

        # --- Step 1: Transcription ---
        transcription_prompt = (
            "Transcribe the provided audio file accurately with timestamps. "
            "Format example:\n[00:00] Speaker: Hello\n[00:05] Speaker: Welcome..."
        )
        transcription_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[transcription_prompt, audio_file]
        )
        transcription = transcription_response.text

        # --- Step 2: Structured extraction ---
        extraction_prompt = f"""
Extract the following information from the conversation text. 
Provide JSON output with keys: RoomType, Cost, Location, StatusOfInhabitant, RequiredAmenities, AlternativeSuggestions.

Conversation Text:
{transcription}
"""
        extraction_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[extraction_prompt]
        )
        structured_text = extraction_response.text

        # --- Step 3: Convert JSON string to Python dict ---
        import json
        try:
            insights = json.loads(structured_text)
        except json.JSONDecodeError:
            # fallback if model doesn't return perfect JSON
            insights = {"raw_extraction": structured_text}

        # --- Step 4: Save to Excel ---
        df_insights = pd.DataFrame({
            "Key": insights.keys(),
            "Value": [str(v) for v in insights.values()]
        })

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_insights.to_excel(writer, index=False, sheet_name="Insights")
            df_transcription = pd.DataFrame({"Transcription": [transcription]})
            df_transcription.to_excel(writer, index=False, sheet_name="Transcription")
        output.seek(0)

        excel_path = os.path.join(temp_dir, f"{file.filename}_insights.xlsx")
        with open(excel_path, "wb") as excel_file:
            excel_file.write(output.getvalue())

        return JSONResponse({
            "transcription": transcription,
            "structured_insights": insights,
            "download_url": f"/download_excel/{file.filename}"
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            if 'client' in locals() and audio_file is not None:
                client.files.delete(name=audio_file.name)
        except Exception:
            pass


@app.get("/download_excel/{filename}")
async def download_excel(filename: str):
    """Download generated Excel file"""
    file_path = os.path.join("temp_audio", f"{filename}_insights.xlsx")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/")
async def root():
    return {"message": "FastAPI - Audio Transcription & Summarization API"}
