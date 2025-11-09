import os
import random
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ML.API.recommend_api import (
    load_orders,
    get_menu,
    get_popular,
    get_highest_rated,
    find_by_category,
    find_item_price
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

try:
    gemini_client = genai.Client()
    MODEL = "gemini-2.5-flash"
except:
    gemini_client = None

def system_prompt():
    m = get_menu()
    p = get_popular(10)
    r = get_highest_rated(10)
    return f"""
You are a smart canteen chatbot. Use the menu, prices, popularity, ratings, and categories to answer accurately.
If asked about a food item, check if it exists in the menu.
Never invent items.
Always keep answers short and friendly.
"""

def rule_reply(msg: str):
    t = msg.lower()

    if "menu" in t:
        m = get_menu()
        if not m:
            return "Menu is unavailable."
        return "\n".join([f"- {i['item_name']} ₹{i['price']} ({i['category']}) rating {round(i['rating'],1)}" for i in m])

    if "popular" in t or "trending" in t:
        p = get_popular(10)
        if not p:
            return "Popularity data unavailable."
        return "\n".join([f"{i+1}. {r['item_name']} score {round(r['score'],1)}" for i, r in enumerate(p)])

    if "highest rated" in t or "best rated" in t:
        r = get_highest_rated(10)
        return "\n".join([f"{i+1}. {row['item_name']} rating {round(row['rating'],1)}" for i, row in enumerate(r)])

    if "price" in t:
        price = find_item_price(msg)
        if price:
            return f"The price is ₹{price}."
        return "I couldn't find that item."

    if "category" in t or "suggest" in t:
        for cat in ["Breakfast", "Lunch", "Snacks", "Beverage"]:
            if cat.lower() in t:
                items = find_by_category(cat)
                if not items:
                    return "No items found."
                return "\n".join([f"{i+1}. {r['item_name']} score {round(r['score'],1)}" for i, r in enumerate(items)])

    if "spicy" in t:
        m = get_menu()
        spicy_items = [i for i in m if i["spicy_level"] >= 3]
        if not spicy_items:
            return "There are no spicy items."
        return "\n".join([f"- {i['item_name']} 🌶️ (level {i['spicy_level']})" for i in spicy_items])

    return None

def is_greeting(msg: str):
    greet = ["hi", "hello", "hey", "yo", "hola", "sup", "hii", "hiii"]
    clean = msg.lower().strip()
    if clean in greet:
        return True
    for g in greet:
        if clean.startswith(g + " "):
            return True
    return False

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    msg = request.new_message

    if is_greeting(msg):
        reply = random.choice([
            "Hey! What’s cooking today? 😄",
            "Hello! Ready for something tasty? 😋",
            "Hi! What can I get you today? 🍽️",
            "Hey there! Hungry for something delicious? 😁"
        ])
        updated = request.history + [
            Content(role="user", parts=[Part(text=msg)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated)

    r = rule_reply(msg)
    if r:
        reply = r
        updated = request.history + [
            Content(role="user", parts=[Part(text=msg)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated)

    if not gemini_client:
        reply = "I'm having trouble connecting to AI. Try asking for menu, price, rating, or category."
        updated = request.history + [
            Content(role="user", parts=[Part(text=msg)]),
            Content(role="model", parts=[Part(text=reply)])
        ]
        return ChatResponse(reply=reply, updated_history=updated)

    prompt = system_prompt()
    convo = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    for h in request.history:
        convo.append(types.Content(role=h.role, parts=[types.Part(text=p.text) for p in h.parts]))

    convo.append(types.Content(role="user", parts=[types.Part(text=msg)]))

    try:
        res = gemini_client.models.generate_content(model=MODEL, contents=convo)
        reply = res.text
    except:
        reply = "Something went wrong while thinking."

    updated = request.history + [
        Content(role="user", parts=[Part(text=msg)]),
        Content(role="model", parts=[Part(text=reply)])
    ]
    return ChatResponse(reply=reply, updated_history=updated)

@router.get("/")
def ping():
    return {"ok": True}
