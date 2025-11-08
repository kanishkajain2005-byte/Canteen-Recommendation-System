import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from google import genai
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------
# ✅ 1. Create FastAPI app
# ------------------------------------------------------
app = FastAPI(
    title="Gemini Conversational API",
    description="A service for multi-turn chat using the Gemini API and persistent history."
)

# ------------------------------------------------------
# ✅ 2. Enable CORS
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# ✅ 3. Load .env
# ------------------------------------------------------
load_dotenv()

# ------------------------------------------------------
# ✅ 4. Pydantic Models
# ------------------------------------------------------
class Part(BaseModel):
    text: str

class Content(BaseModel):
    role: str = Field(..., pattern="^(user|model)$")
    parts: List[Part]

class ChatRequest(BaseModel):
    history: List[Content]
    new_message: str

class ChatResponse(BaseModel):
    reply: str
    updated_history: List[Content]

# ------------------------------------------------------
# ✅ 5. Initialize Gemini Client
# ------------------------------------------------------
try:
    gemini_client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    gemini_client = None
    print("CRITICAL: Failed to initialize Gemini Client:", e)

# ------------------------------------------------------
# ✅ 6. Chat Endpoint (FIXED ✅ WORKING ✅)
# ------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    if not gemini_client:
        raise HTTPException(503, "AI service unavailable")

    # ✅ Build conversation in the NEW correct GenAI JSON format
    conversation = []

    for msg in request.history:
        conversation.append({
            "role": msg.role,
            "parts": [{"text": p.text} for p in msg.parts]
        })

    # ✅ Add new message
    conversation.append({
        "role": "user",
        "parts": [{"text": request.new_message}]
    })

    try:
        # ✅ Correct Gemini API Call
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation
        )

        reply_text = response.text

        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply_text)])
        ]

        return ChatResponse(
            reply=reply_text,
            updated_history=updated_history
        )

    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")

# ------------------------------------------------------
# ✅ 7. Test Endpoint
# ------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Conversational Chat API is running!"}
