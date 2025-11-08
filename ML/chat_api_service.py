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

app = FastAPI(
    title="Canteen Chatbot API",
    description="Dynamic canteen chatbot with menu, stock, specials, sentiment & recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MENU_PATH = "ML/Data/raw/menu.csv"


def load_menu():
    df = pd.read_csv(MENU_PATH)
    df["available"] = df["available"].astype(str).str.lower().isin(["yes", "true", "1"])
    return df


MENU_DF = load_menu()


def menu_to_text(df):
    return "\n".join([f"- {row['item_name']} — ₹{row['price']}" for _, row in df.iterrows()])


CANTEEN_MENU_TEXT = menu_to_text(MENU_DF)

DAILY_STOCK = {row["item_name"]: row["available"] for _, row in MENU_DF.iterrows()}

DAILY_SPECIALS = MENU_DF.sample(2)["item_name"].tolist()

ORDER_DATA = load_orders()
POPULARITY_DATA = get_popular(ORDER_DATA, top_n=10)
POPULARITY_RANK = {entry["item_name"]: idx + 1 for idx, entry in enumerate(POPULARITY_DATA)}


def detect_mood(text):
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
except Exception:
    gemini_client = None


def build_system_prompt(user_message):
    mood = detect_mood(user_message)

    mood_hint = {
        "tired": "User is tired. Suggest energy boosters like Cold Coffee.",
        "hungry": "User is extremely hungry. Suggest filling meals like Veg Thali or Paneer Thali.",
        "sad": "User is sad. Suggest mood-lifting comfort foods like Maggi or Samosa.",
        "angry": "User is irritated. Suggest quick-served food like Samosa.",
    }.get(mood, "")

    return f"""
You are the official Canteen AI Assistant.

MENU:
{CANTEEN_MENU_TEXT}

STOCK STATUS:
{DAILY_STOCK}

POPULAR ITEMS:
{POPULARITY_RANK}

TODAY'S SPECIALS:
{DAILY_SPECIALS}

MOOD HINT:
{mood_hint}

BEHAVIOR RULES:
- For greetings like "hi", "hello", "how are you", respond friendly & conversational.
- For dish availability: check stock and respond accordingly.
- For out of stock items: say they are currently unavailable.
- For recommendations: reply EXACTLY this JSON:
  {{"action": "recommend", "query": "<user message>"}}
- If a user asks about an item not in the menu: respond "Sorry, that item is not available in our canteen."
- Never invent items or prices.
- Keep responses helpful, natural, friendly, and human-like.
"""


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    if not gemini_client:
        raise HTTPException(503, "AI unavailable")

    system_prompt = build_system_prompt(request.new_message)

    conversation = [
        types.Content(role="user", parts=[types.Part(text=system_prompt)])
    ]

    for msg in request.history:
        conversation.append(
            types.Content(
                role=msg.role,
                parts=[types.Part(text=p.text) for p in msg.parts]
            )
        )

    conversation.append(
        types.Content(
            role="user",
            parts=[types.Part(text=request.new_message)]
        )
    )

    try:
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=conversation
        )

        reply = response.text

        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]

        return ChatResponse(reply=reply, updated_history=updated_history)

    except Exception as e:
        raise HTTPException(500, f"Gemini API Error: {e}")


@app.get("/")
def home():
    return {"msg": "Dynamic Canteen Chatbot ✅ Running!"}
