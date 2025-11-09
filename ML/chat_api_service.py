import os
import random
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from ML.API.recommend_api import (
    load_orders,
    get_menu,
    get_popular,
    get_highest_rated,
    find_by_category,
)

load_dotenv()

router = APIRouter(prefix="/chat", tags=["chat"])

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

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except:
        return None

def _menu_text():
    m = _safe(get_menu) or []
    if not m:
        return "Menu unavailable."
    return "\n".join([f"- {row.get('item_name')} — ₹{row.get('price')} ({row.get('category')})" for row in m])

def _popular_text():
    p = _safe(get_popular, 10) or []
    if not p:
        return "No popularity data."
    return "\n".join([f"{i+1}. {r.get('item_name')} — score {r.get('score')}" for i, r in enumerate(p)])

def _rated_text():
    r = _safe(get_highest_rated, 10) or []
    if not r:
        return "No rating data."
    return "\n".join([
        f"{i+1}. {row.get('item_name')} — avg rating {round(float(row.get('rating', 0)),2)}"
        for i, row in enumerate(r)
    ])

def _category_text(cat: str):
    items = _safe(find_by_category, cat, 10) or []
    if not items:
        return f"No items found for category '{cat}'."
    return "\n".join([
        f"{i+1}. {r.get('item_name')} — score {round(float(r.get('score', 0)),2)}"
        for i, r in enumerate(items)
    ])

def _rule_based_reply(text: str):
    t = text.lower()
    if "menu" in t or "list items" in t:
        return f"Here is the menu:\n{_menu_text()}"
    if "popular" in t or "trending" in t:
        return f"Top popular picks:\n{_popular_text()}"
    if "highest rated" in t or "high rating" in t or "best rated" in t:
        return f"Highest rated dishes:\n{_rated_text()}"
    if "category" in t and "suggest" in t:
        parts = t.split("category")
        cat = parts[-1].strip(": ,.")
        return f"Top items in category '{cat}':\n{_category_text(cat)}"
    if "breakfast" in t:
        return f"Breakfast ideas:\n{_category_text('Breakfast')}"
    if "lunch" in t:
        return f"Lunch ideas:\n{_category_text('Lunch')}"
    if "snack" in t:
        return f"Snack ideas:\n{_category_text('Snacks')}"
    return "I can show the menu, popular items, highest-rated dishes, or category suggestions. Try: 'show menu', 'popular items', 'highest rated', or 'suggest category Lunch'."

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    msg = request.new_message.lower().strip()
    greetings = ["hi", "hello", "hey", "yo", "sup", "hii", "hiii", "hola"]

    if msg in greetings or any(msg.startswith(g) for g in greetings):
        reply = random.choice([
            "Hey! What’s cookin'? 😊",
            "Hello! Hungry for something tasty? 😄",
            "Hi! What can I get you today? 🍽️",
            "Hey there! Feeling hungry? 😁"
        ])
        updated = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated)

    reply = _rule_based_reply(request.new_message)

    updated = request.history + [
        Content(role="user", parts=[Part(text=request.new_message)]),
        Content(role="model", parts=[Part(text=reply)])
    ]
    return ChatResponse(reply=reply, updated_history=updated)

@router.get("/")
def ping():
    return {"ok": True, "hint": "POST /chat/chat with {history, new_message}"}
