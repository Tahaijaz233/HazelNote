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
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI()
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

class VideoRequest(BaseModel):
    url: str

class ManualRequest(BaseModel):
    text: str

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

def get_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_transcript(video_url):
    print(f"📥 Fetching YouTube: {video_url}")
    video_id = get_video_id(video_url)
    
    # 1. Try API (Fastest)
    if video_id:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return clean_text(" ".join([item['text'] for item in transcript_list]))
        except: pass

    # 2. Try yt-dlp (iOS Client - often less blocked)
    try:
        if os.path.exists("transcript.en.vtt"): os.remove("transcript.en.vtt")
        cmd = [
            "yt-dlp", "--write-auto-sub", "--skip-download", 
            "--sub-lang", "en", 
            "--extractor-args", "youtube:player_client=ios", 
            "--output", "transcript", 
            video_url
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists("transcript.en.vtt"):
            with open("transcript.en.vtt", "r", encoding="utf-8") as f: content = f.read()
            os.remove("transcript.en.vtt")
            lines = [l.strip() for l in content.splitlines() if "-->" not in l and l.strip() and not l.startswith(("WEBVTT", "Kind:", "Language:"))]
            return clean_text(" ".join(list(dict.fromkeys(lines))))
    except: pass
    
    return None

def extract_pdf_text(file_path):
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return clean_text(text)
    except: return None

def try_generate(client, prompt):
    model = "gemini-2.5-flash" 
    try:
        response = client.models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(max_output_tokens=8192)
        )
        return response.text
    except Exception as e: return f"Error: {str(e)}"

# --- BRAIN LOGIC ---
def generate_study_data(client, text):
    print("...Generating Content...")
    
    # NOTES PROMPT
    notes_prompt = """
    You are an expert academic tutor.
    1. Write a SUMMARY (250 words). Use `$$x^2$$` for all math equations.
    2. Write "===SPLIT==="
    3. Write DETAILED NOTES (Markdown).
       - Use ## for Topics.
       - **Bold** key terms.
       - Math: ALWAYS use `$$` for block math and `$` for inline. Example: $$E=mc^2$$.
       - **DIAGRAMS:** Use Mermaid.js. Quote strings!
         ```mermaid
         graph TD; A["Start"] --> B["End"];
         ```
    """
    raw_notes = try_generate(client, f"{notes_prompt}\n\nCONTENT:\n{text[:60000]}")
    if "Error:" in raw_notes: return {"error": raw_notes}
    
    parts = raw_notes.split("===SPLIT===")
    summary = parts[0].strip()
    notes_text = parts[1].strip() if len(parts) > 1 else raw_notes

    # EXTRAS PROMPT
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
                    p = line.split("|")
                    if len(p) >= 2: flashcards.append({"front": p[0].replace("Front:", "").strip(), "back": p[1].replace("Back:", "").strip()})
        except: pass

    quiz = []
    if "===QUIZ===" in raw_extras:
        try:
            for line in raw_extras.split("===QUIZ===")[1].splitlines():
                if "|" in line and "Q:" in line:
                    p = line.split("|")
                    if len(p) >= 5:
                        quiz.append({"question": p[0].replace("Q:", "").strip(), "options": [x.strip() for x in p[1:-1]], "answer": p[-1].replace("Answer:", "").strip()})
        except: pass

    # PODCAST PROMPT
    podcast_prompt = """
    Convert this content into a teaching monologue.
    - Role: Narrator speaking to a student.
    - Tone: Engaging, clear, storytelling.
    - Math: Explain concepts in words, but keep `$$formula$$` in the text for display.
    - End with "Any questions?".
    """
    podcast_script = try_generate(client, f"{podcast_prompt}\n\nCONTENT:\n{summary}")

    return {"summary": summary, "notes": notes_text, "flashcards": flashcards, "quiz": quiz, "podcast": podcast_script, "raw_transcript": text[:20000]}

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze_video")
async def analyze_video(req: VideoRequest, x_gemini_api_key: str = Header(None)):
    client = get_client(x_gemini_api_key)
    transcript = get_transcript(req.url)
    if not transcript: 
        # Return 422 to trigger the "Manual Input" modal on frontend
        return JSONResponse(content={"error": "YouTube blocked the request. Use the 'Paste Text' option."}, status_code=422)
    return generate_study_data(client, transcript)

@app.post("/api/analyze_text")
async def analyze_text(req: ManualRequest, x_gemini_api_key: str = Header(None)):
    client = get_client(x_gemini_api_key)
    return generate_study_data(client, req.text)

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
    uvicorn.run(app, host="0.0.0.0", port=7860)
