from typing import List, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel, Field
import re

from ML.API.recommend_api import (
    get_menu,
    get_popular,
    get_highest_rated,
    find_by_category,
)

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

def _lower(s: str):
    return re.sub(r"\s+", " ", s.strip().lower())

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except:
        return None

def _menu_list() -> List[Dict[str, Any]]:
    m = _safe(get_menu) or []
    out = []
    for row in m:
        if row.get("item_name") and isinstance(row.get("price"), (int, float)):
            out.append({
                "item_name": row["item_name"],
                "price": float(row["price"]),
                "category": row.get("category")
            })
    return out

def _fmt(rows: List[Dict[str, Any]]):
    if not rows:
        return "No items found."
    lines = []
    for i, r in enumerate(rows):
        base = f"{i+1}. {r['item_name']} — ₹{int(r['price']) if float(r['price']).is_integer() else r['price']}"
        if r.get("category"):
            base += f" ({r['category']})"
        lines.append(base)
    return "\n".join(lines)

def _spicy_filter(menu: List[Dict[str, Any]]):
    spicy_words = ["spicy", "masala", "chilli", "hot", "tandoori"]
    out = []
    for r in menu:
        name = _lower(r["item_name"])
        if any(w in name for w in spicy_words):
            out.append(r)
    return out

def _budget(text: str):
    m = re.search(r"(?:under|below|upto|up to|<=)\s*₹?\s*(\d+)", text)
    if m:
        return float(m.group(1))
    return None

def _category(text: str):
    m = re.search(r"category\s+([a-zA-Z ]+)", text)
    if m:
        return m.group(1).strip().title()
    return None

def _reply(text: str) -> str:
    t = _lower(text)
    menu = _menu_list()

    greetings = ["hi", "hello", "hey", "hola", "namaste"]
    if any(t == g or t.startswith(g + " ") for g in greetings):
        return "Hey! What can I get you today? 🙂"

    if "menu" in t:
        return "Here is the menu:\n" + _fmt(menu)

    if "popular" in t or "trending" in t:
        p = _safe(get_popular, 10) or []
        return "Popular items:\n" + _fmt(p)

    if "highest rated" in t or "top rated" in t or "best rated" in t:
        r = _safe(get_highest_rated, 10) or []
        return "Highest-rated dishes:\n" + _fmt(r)

    if "spicy" in t or "something hot" in t:
        spicy = _spicy_filter(menu)
        return "Spicy dishes:\n" + _fmt(spicy)

    b = _budget(t)
    if b:
        affordable = [r for r in menu if r["price"] <= b]
        return f"Items under ₹{int(b)}:\n" + _fmt(affordable)

    cat = _category(t)
    if cat:
        items = _safe(find_by_category, cat, 20) or []
        return f"Items in category '{cat}':\n" + _fmt(items)

    exact = [r for r in menu if _lower(r["item_name"]) == t]
    if exact:
        r = exact[0]
        return f"{r['item_name']} costs ₹{int(r['price'])}."

    return "I can show the menu, popular items, highest-rated dishes, spicy foods, or category-wise suggestions. Try: 'show menu', 'popular items', 'spicy items', 'highest rated', or 'category snacks'."

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply = _reply(request.new_message)
    updated = request.history + [
        Content(role="user", parts=[Part(text=request.new_message)]),
        Content(role="model", parts=[Part(text=reply)])
    ]
    return ChatResponse(reply=reply, updated_history=updated)

@router.get("/")
def ping():
    return {"ok": True}
