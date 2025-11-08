import os
import random
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from ML.API.recommend_api import load_orders, get_popular

load_dotenv()

app = FastAPI(title="Canteen Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MENU_PATH = "ML/Data/raw/menu.csv"

def lazy_menu():
    df = pd.read_csv(MENU_PATH)
    df["available"] = df["available"].astype(str).str.lower().isin(["yes", "true", "1"])
    return df

def menu_text():
    df = lazy_menu()
    return "\n".join([f"- {row['item_name']} — ₹{row['price']}" for _, row in df.iterrows()])

def stock_status():
    df = lazy_menu()
    return {row["item_name"]: row["available"] for _, row in df.iterrows()}

def specials():
    df = lazy_menu()
    return df.sample(min(2, len(df)))["item_name"].tolist()

def popularity_rank():
    orders = load_orders()
    pop = get_popular(orders, top_n=10)
    return {entry["item_name"]: i + 1 for i, entry in enumerate(pop)}

def mood_detect(text):
    t = text.lower()
    if any(k in t for k in ["tired", "sleepy"]): return "tired"
    if any(k in t for k in ["sad", "upset"]): return "sad"
    if any(k in t for k in ["angry", "irritated"]): return "angry"
    if any(k in t for k in ["hungry", "starving"]): return "hungry"
    return None

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

try:
    gemini_client = genai.Client()
    MODEL = "gemini-2.5-flash"
except:
    gemini_client = None

def system_prompt(user_message):
    m = mood_detect(user_message)
    menu = menu_text()
    stock = stock_status()
    pop = popularity_rank()
    sp = specials()
    mh = {
        "tired": "User is tired. Suggest energy boosters like Cold Coffee.",
        "hungry": "User is hungry. Suggest Veg Thali or Paneer Thali.",
        "sad": "User is sad. Suggest comfort foods like Maggi or Samosa.",
        "angry": "User is irritated. Suggest quick items like Samosa."
    }.get(m, "")
    return f"""
You are the canteen assistant.

MENU:
{menu}

STOCK:
{stock}

POPULAR:
{pop}

SPECIALS:
{sp}

MOOD:
{mh}

RULES:
- For greetings respond friendly.
- For availability check stock strictly.
- For unavailable items say they are not available.
- For recommendations respond exactly in JSON:
  {{"action":"recommend","query":"<user message>"}}
- Never invent items.
"""

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not gemini_client:
        raise HTTPException(503, "AI unavailable")

    sp = system_prompt(request.new_message)
    convo = [types.Content(role="user", parts=[types.Part(text=sp)])]

    for msg in request.history:
        convo.append(
            types.Content(
                role=msg.role,
                parts=[types.Part(text=p.text) for p in msg.parts]
            )
        )

    convo.append(
        types.Content(role="user", parts=[types.Part(text=request.new_message)])
    )

    try:
        res = gemini_client.models.generate_content(model=MODEL, contents=convo)
        reply = res.text
        updated = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/")
def root():
    return {"msg": "Chatbot Live ✅"}
