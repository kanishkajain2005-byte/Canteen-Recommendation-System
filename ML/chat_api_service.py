import os
import requests
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
# ✅ 2. Canteen Menu (edit anytime)
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
You are the AI assistant for the College Canteen.

Your rules:
1. Only use the following menu:
{CANTEEN_MENU}

2. If the user asks for recommendations (like spicy items, healthy dishes, what to eat, best options, etc.)
   respond ONLY with:
   {{"action": "recommend", "query": "<user message>"}}

3. For questions about menu, prices, what is available:
   - Answer briefly using only the menu.

4. If an item is not in the menu:
   - Reply: "Sorry, that item is not available in our canteen."

5. Do NOT invent new dishes.

6. Keep responses friendly.
"""

# ---------------------------------------------------------
# ✅ Pydantic Models
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
# ✅ Initialize Gemini Client
# ---------------------------------------------------------
try:
    gemini_client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    print("Gemini initialization failed:", e)
    gemini_client = None

# ---------------------------------------------------------
# ✅ Helper: Call Recommendation API
# ---------------------------------------------------------
def call_recommendation_api(query: str):
    RECOMMEND_URL = "http://127.0.0.1:8000/recommend/predict"   # LOCAL
    # For Render:
    # RECOMMEND_URL = "https://canteen-recommendation-system.onrender.com/recommend/predict"

    try:
        res = requests.post(RECOMMEND_URL, json={"query": query})
        if res.status_code != 200:
            return None
        
        return res.json()["recommendations"]
    
    except Exception as e:
        print("Recommendation API error:", e)
        return None

# ---------------------------------------------------------
# ✅ 5. Chat Endpoint (Main Logic)
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    if not gemini_client:
        raise HTTPException(503, "AI service unavailable")

    conversation = []

    # ✅ Add system prompt FIRST
    conversation.append(
        types.Content(role="system", parts=[types.Part(text=SYSTEM_PROMPT)])
    )

    # ✅ Add chat history
    for msg in request.history:
        conversation.append(
            types.Content(
                role=msg.role,
                parts=[types.Part(text=p.text) for p in msg.parts]
            )
        )

    # ✅ Add new user message
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=request.new_message)])
    )

    try:
        # ✅ Call Gemini
        ai_response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation
        )

        reply_text = ai_response.text.strip()

        # ✅ Detect recommendation trigger
        if reply_text.startswith("{") and reply_text.endswith("}") and '"action": "recommend"' in reply_text:
            import json
            parsed = json.loads(reply_text)
            user_query = parsed["query"]

            # ✅ Call your recommendation ML API
            recos = call_recommendation_api(user_query)

            if recos:
                pretty = "\n".join([f"• {item}" for item in recos])
                final_reply = f"Here are some recommendations based on what you said:\n{pretty}"
            else:
                final_reply = "Sorry, I couldn't fetch recommendations right now."

        else:
            # ✅ Normal answer
            final_reply = reply_text

        # ✅ Build new history
        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=final_reply)])
        ]

        return ChatResponse(
            reply=final_reply,
            updated_history=updated_history
        )

    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")

# ---------------------------------------------------------
# ✅ 6. Test endpoint
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Canteen Chatbot API with Recommendations ✅", "docs": "/chat/docs"}
