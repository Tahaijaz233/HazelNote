import os
import subprocess
import re
from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from google.genai import types
import pypdf

app = FastAPI()
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

class VideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    context: str
    question: str
    full_content: str = ""

# --- HELPERS ---
def get_client(api_key: str):
    if not api_key or api_key == "null" or len(api_key) < 10:
        raise HTTPException(status_code=401, detail="Please enter your Gemini API Key in Settings ⚙️")
    return genai.Client(api_key=api_key)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def get_transcript(video_url):
    print(f"📥 Fetching YouTube: {video_url}")
    try:
        if os.path.exists("transcript.en.vtt"): os.remove("transcript.en.vtt")
        cmd = ["yt-dlp", "--write-auto-sub", "--skip-download", "--sub-lang", "en", "--output", "transcript", video_url]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists("transcript.en.vtt"):
            with open("transcript.en.vtt", "r", encoding="utf-8") as f: content = f.read()
            os.remove("transcript.en.vtt")
            lines = [l.strip() for l in content.splitlines() if "-->" not in l and l.strip() and not l.startswith(("WEBVTT", "Kind:", "Language:"))]
            return clean_text(" ".join(list(dict.fromkeys(lines))))
        return None
    except Exception as e:
        print(f"YouTube Error: {e}")
        return None

def extract_pdf_text(file_path):
    print(f"📄 Reading PDF: {file_path}")
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return clean_text(text)
    except Exception as e:
        return None

def try_generate(client, prompt):
    model = "gemini-2.5-flash" 
    try:
        print(f"🧠 Gemini 2.5 Flash is thinking...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=8192)
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- BRAIN LOGIC ---
def generate_study_data(client, text):
    print("...Generating Content...")
    
    # 1. NOTES
    notes_prompt = """
    You are an expert academic tutor.
    1. Write a SUMMARY (250 words). 
       - CRITICAL: Use `$$...$$` for block math and `$...$` for inline math.
    2. Write "===SPLIT==="
    3. Write DETAILED NOTES (Markdown).
       - Use ## for Topics and ### for Sub-topics.
       - **Bold** key terms.
       - Math: Use `$$...$$` for equations.
       - **DIAGRAMS:** Use Mermaid.js. 
       - **CRITICAL:** Use simple node names (A, B, C) and put text in quotes.
         ```mermaid
         graph TD;
            A["Concept Start"] --> B["Process Step"];
            B --> C["Result"];
         ```
    """
    raw_notes = try_generate(client, f"{notes_prompt}\n\nCONTENT:\n{text[:60000]}")
    if "Error:" in raw_notes: return {"error": raw_notes}
    parts = raw_notes.split("===SPLIT===")
    summary = parts[0].strip()
    notes_text = parts[1].strip() if len(parts) > 1 else raw_notes

    # 2. EXTRAS
    extras_prompt = """
    Create study aids.
    ===FLASHCARDS===
    Front: [Term] | Back: [Def]
    ===QUIZ===
    Q: [Question] | A: [Opt1] | B: [Opt2] | C: [Opt3] | Answer: [Full Answer Text]
    """
    raw_extras = try_generate(client, f"{extras_prompt}\n\nCONTENT:\n{text[:40000]}")
    
    flashcards = []
    if "===FLASHCARDS===" in raw_extras:
        try:
            for line in raw_extras.split("===FLASHCARDS===")[1].split("===QUIZ===")[0].splitlines():
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2: flashcards.append({"front": parts[0].replace("Front:", "").strip(), "back": parts[1].replace("Back:", "").strip()})
        except: pass

    quiz = []
    if "===QUIZ===" in raw_extras:
        try:
            for line in raw_extras.split("===QUIZ===")[1].splitlines():
                if "|" in line and "Q:" in line:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        quiz.append({
                            "question": parts[0].replace("Q:", "").strip(), 
                            "options": [p.strip() for p in parts[1:-1]], 
                            "answer": parts[-1].replace("Answer:", "").strip()
                        })
        except: pass

    # 3. PODCAST SCRIPT
    podcast_prompt = """
    Convert this content into a teaching monologue. 
    - You are the Narrator speaking directly to a student.
    - Explain math concepts conceptually in spoken English, but use LaTeX `$$x$$` for the transcript.
    - Break the explanation into short, digestible paragraphs (3-4 sentences max).
    - End the script with "Any questions?".
    """
    podcast_script = try_generate(client, f"{podcast_prompt}\n\nCONTENT:\n{summary}")

    return {
        "summary": summary, "notes": notes_text, 
        "flashcards": flashcards, "quiz": quiz, 
        "podcast": podcast_script, "raw_transcript": text[:20000]
    }

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze_video")
async def analyze_video(req: VideoRequest, x_gemini_api_key: str = Header(None)):
    client = get_client(x_gemini_api_key)
    transcript = get_transcript(req.url)
    if not transcript: return JSONResponse(content={"error": "No transcript found."}, status_code=400)
    return generate_study_data(client, transcript)

@app.post("/api/analyze_pdf")
async def analyze_pdf(x_gemini_api_key: str = Header(None), file: UploadFile = File(...)):
    client = get_client(x_gemini_api_key)
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer: buffer.write(await file.read())
    text = extract_pdf_text(temp_filename)
    if os.path.exists(temp_filename): os.remove(temp_filename)
    if not text: return JSONResponse(content={"error": "Could not read PDF."}, status_code=400)
    return generate_study_data(client, text)

@app.post("/api/chat")
async def chat_with_context(req: ChatRequest, x_gemini_api_key: str = Header(None)):
    client = get_client(x_gemini_api_key)
    return {"answer": try_generate(client, f"Tutor Mode. Context: {req.full_content[:5000]}. Question: {req.question}")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)