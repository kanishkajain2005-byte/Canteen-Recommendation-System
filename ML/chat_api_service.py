import random
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

from ML.API.recommend_api import (
    load_dataset,
    get_menu,
    get_popular,
    get_highest_rated,
    find_by_category,
    spicy_items
)

router = APIRouter(prefix="/chat", tags=["chat"])

DATA_CACHE = load_dataset()
MENU_CACHE = get_menu()

client = genai.Client()
MODEL = "gemini-2.5-flash"

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

def build_system_instruction():
    menu = MENU_CACHE
    popular = get_popular(10)
    rated = get_highest_rated(10)
    spicy = spicy_items()[:10]

    menu_lines = [
        f"- {m['item_name']} (₹{m['price']}) | Category: {m['category']} | Rating: {m.get('rating', 'N/A')}"
        for m in menu
    ]

    pop_lines = [
        f"{i+1}. {p['item_name']} — Popularity Score: {p.get('popularity_score', 0)}"
        for i, p in enumerate(popular)
    ]

    rated_lines = [
        f"{i+1}. {r['item_name']} — Rating: {r.get('rating', 0):.1f}/5"
        for i, r in enumerate(rated)
    ]

    spicy_lines = [
        f"{i+1}. {s['item_name']} — Spice Level: {s.get('spicy_level', 0)}"
        for i, s in enumerate(spicy)
    ]

    return f"""
You are the official canteen chatbot.

Rules:
- Always answer using ONLY the information provided below.
- Never invent prices or items.
- Always use INR currency (₹). Never show $, USD, or convert currency.
- If user asks something not in the menu, say it is not available.
- Be friendly and conversational, but accurate.
- When asked about price, say: "₹<amount>".

MENU:
{chr(10).join(menu_lines)}

POPULAR ITEMS:
{chr(10).join(pop_lines)}

HIGHEST RATED:
{chr(10).join(rated_lines)}

SPICY ITEMS:
{chr(10).join(spicy_lines)}
"""

def is_greeting(text: str):
    t = text.lower().strip()
    words = t.split()

    greetings = {"hi", "hello", "hey", "hola", "yo", "sup"}

    return len(words) == 1 and words[0] in greetings

def greeting_reply():
    return random.choice([
        "Hey! What can I get you today? 😊",
        "Hello! Hungry for something yummy? 🍽️",
        "Hi! What are you craving today? 😄",
        "Hey there! Looking for something tasty? 😁"
    ])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if is_greeting(request.new_message):
        reply = greeting_reply()
        updated = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated)

    prompt = build_system_instruction()

    convo = [types.Content(role="user", parts=[types.Part(text=prompt)])]

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
        res = client.models.generate_content(
            model=MODEL,
            contents=convo
        )
        reply = res.text
    except Exception as e:
        raise HTTPException(500, str(e))

    updated = request.history + [
        Content(role="user", parts=[Part(text=request.new_message)]),
        Content(role="model", parts=[Part(text=reply)])
    ]

    return ChatResponse(reply=reply, updated_history=updated)

@router.get("/")
def ping():
    return {"ok": True, "message": "Chatbot API online"}
