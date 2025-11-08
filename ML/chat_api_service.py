import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# ✅ 1. FastAPI App + CORS
# ---------------------------------------------------------
app = FastAPI(
    title="Gemini Conversational API",
    description="Canteen-aware chatbot with menu + recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# ---------------------------------------------------------
# ✅ 2. Canteen Menu (You can edit this anytime)
# ---------------------------------------------------------
CANTEEN_MENU = """
Available items in our University Canteen:

- Veg Sandwich — ₹40  
- Cheese Maggi — ₹60  
- Veg Thali — ₹80  
- Paneer Thali — ₹110  
- Cold Coffee — ₹50  
- Masala Dosa — ₹70  
- Samosa — ₹15  
- Chole Bhature — ₹65  
- Idli Sambhar — ₹40  
- Fried Rice — ₹70  
"""

SYSTEM_PROMPT = f"""
You are the official AI assistant for the College Canteen.

Your rules:
1. You ONLY use the following menu and nothing outside it:
{CANTEEN_MENU}

2. If a user asks for recommendations, spicy food, healthy items, popular items, or "what should I eat?", 
   you MUST respond with this JSON EXACTLY:
   {{"action": "recommend", "query": "<user message>"}}
   (No additional text.)

3. If a user asks general questions about the canteen (timings, menu items, prices):
   - Answer briefly and factually based only on the menu above.

4. If a user asks about a food item NOT in the menu:
   - Respond: "Sorry, that item is not available in our canteen."

5. Never hallucinate new dishes or ingredients.

6. Always stay friendly and helpful.
"""

# ---------------------------------------------------------
# ✅ 3. Pydantic Models
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# ✅ 4. Initialize Gemini Client
# ---------------------------------------------------------
try:
    gemini_client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    print("Gemini initialization failed:", e)
    gemini_client = None

# ---------------------------------------------------------
# ✅ 5. Chat Endpoint
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    if not gemini_client:
        raise HTTPException(503, "AI service unavailable")

    conversation = []

    # ✅ Inject system prompt FIRST
    conversation.append(
        types.Content(
            role="system",
            parts=[types.Part.from_text(SYSTEM_PROMPT)]
        )
    )

    # ✅ Convert past messages
    for msg in request.history:
        conversation.append(
            types.Content(
                role=msg.role,
                parts=[types.Part.from_text(p.text) for p in msg.parts]
            )
        )

    # ✅ Add new user message
    conversation.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(request.new_message)]
        )
    )

    try:
        # ✅ Call Gemini
        ai_response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation
        )

        reply_text = ai_response.text

        # ✅ Update history
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


# ---------------------------------------------------------
# ✅ 6. Test endpoint
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Canteen Chatbot API running ✅", "docs": "/chat/docs"}
