import re
from typing import List, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

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

def _lower(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None

def _menu_rows() -> List[Dict[str, Any]]:
    m = _safe(get_menu) or []
    rows = []
    for row in m:
        name = str(row.get("item_name", "")).strip()
        price = row.get("price", None)
        cat = str(row.get("category", "")).strip()
        if name and isinstance(price, (int, float)):
            rows.append({"item_name": name, "price": float(price), "category": cat})
    return rows

def _format_list(rows: List[Dict[str, Any]], limit: int = 10, with_score: bool = False, with_rating: bool = False) -> str:
    if not rows:
        return "No items found."
    out = []
    for i, r in enumerate(rows[:limit]):
        base = f"{i+1}. {r.get('item_name')}"
        if with_score and r.get("score") is not None:
            base += f" — score {round(float(r['score']),2)}"
        if with_rating and r.get("rating") is not None:
            base += f" — avg rating {round(float(r['rating']),2)}"
        if r.get("price") is not None:
            base += f" — ₹{int(r['price']) if float(r['price']).is_integer() else round(float(r['price']),2)}"
        if r.get("category"):
            base += f" ({r['category']})"
        out.append(base)
    return "\n".join(out)

def _best_effort_item_lookup(query: str, menu: List[Dict[str, Any]]):
    q = _lower(query)
    exact = [r for r in menu if _lower(r["item_name"]) == q]
    if exact:
        return exact[0]
    contains = [r for r in menu if q in _lower(r["item_name"])]
    if contains:
        return contains[0]
    return None

def _extract_budget(text: str):
    m = re.search(r"(?:under|below|<=?|max|upto|up to)\s*₹?\s*(\d+)", text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"₹\s*(\d+)", text)
    if m2 and any(k in text for k in ["under", "below", "<=", "max", "upto", "up to"]):
        return float(m2.group(1))
    return None

def _detect_category(text: str):
    cat_words = ["breakfast", "lunch", "dinner", "snack", "snacks", "beverage", "beverages", "drinks", "dessert", "veg", "non-veg", "north indian", "south indian", "chinese", "starter", "starters"]
    for w in cat_words:
        if w in text:
            return w.title() if w not in ["non-veg"] else "Non-Veg"
    if "category" in text:
        tail = _lower(text.split("category", 1)[-1]).strip(" :-,.'\"")
        if tail:
            return tail.title()
    return None

def _mood_suggest(text: str, menu: List[Dict[str, Any]]):
    t = text
    mood = None
    if any(k in t for k in ["tired", "sleepy", "exhausted"]): mood = "tired"
    elif any(k in t for k in ["sad", "upset", "down"]): mood = "sad"
    elif any(k in t for k in ["angry", "irritated", "annoyed"]): mood = "angry"
    elif any(k in t for k in ["hungry", "starving", "famished"]): mood = "hungry"
    if not mood:
        return None
    picks = []
    def pick_contains(words):
        for r in menu:
            n = _lower(r["item_name"])
            if any(w in n for w in words):
                picks.append(r)
                if len(picks) >= 3:
                    return True
        return False
    if mood == "tired":
        pick_contains(["coffee", "cold coffee", "tea"])
    elif mood == "sad":
        pick_contains(["magg", "samosa", "pasta", "noodles"])
    elif mood == "angry":
        pick_contains(["samosa", "roll", "fries", "cutlet"])
    elif mood == "hungry":
        pick_contains(["thali", "biryani", "paneer", "rice"])
    if not picks:
        return None
    return "Here are a few picks to match your mood:\n" + _format_list(picks, limit=3)

def _rule_engine(user_text: str) -> str:
    text = _lower(user_text)
    greetings = ["hi", "hello", "hey", "yo", "sup", "hii", "hiii", "hola", "namaste"]
    if any(text == g or text.startswith(g + " ") for g in greetings):
        return "hey! what can i get you today? 🙂"
    menu = _menu_rows()
    mood_hint = _mood_suggest(text, menu)
    if any(k in text for k in ["menu", "show menu", "list items", "what do you have"]):
        return "here’s the current menu:\n" + _format_list(menu, limit=25)
    if any(k in text for k in ["popular", "trending", "most ordered", "top picks"]):
        pop = _safe(get_popular, 10) or []
        return "top popular picks right now:\n" + _format_list(pop, limit=10, with_score=True)
    if any(k in text for k in ["highest rated", "high rating", "best rated", "top rated"]):
        rated = _safe(get_highest_rated, 10) or []
        return "highest-rated dishes:\n" + _format_list(rated, limit=10, with_rating=True)
    budget = _extract_budget(text)
    if budget is not None or any(k in text for k in ["cheap", "affordable", "budget"]):
        cap = budget if budget is not None else 120.0
        affordable = [r for r in menu if r.get("price") is not None and float(r["price"]) <= cap]
        if not affordable:
            return f"couldn’t find items under ₹{int(cap)}."
        pop = _safe(get_popular, 50) or []
        pop_names = [_lower(p.get("item_name","")) for p in pop]
        affordable_sorted = sorted(affordable, key=lambda r: (0 if _lower(r["item_name"]) in pop_names else 1, r["price"]))
        header = f"nice! here are tasty picks under ₹{int(cap)}:\n"
        return header + _format_list(affordable_sorted, limit=10)
    cat = _detect_category(text)
    if cat:
        items = _safe(find_by_category, cat, 15) or []
        if items:
            return f"here are some picks in '{cat}':\n" + _format_list(items, limit=10, with_score=True)
    name_match = None
    tokens = re.findall(r"[a-zA-Z][a-zA-Z ]{1,40}", text)
    for tok in sorted(set(tokens), key=len, reverse=True):
        hit = _best_effort_item_lookup(tok, menu)
        if hit:
            name_match = hit
            break
    if name_match:
        base = f"{name_match['item_name']} costs ₹{int(name_match['price']) if float(name_match['price']).is_integer() else round(float(name_match['price']),2)}"
        if name_match.get("category"):
            base += f" and is in {name_match['category']}."
        else:
            base += "."
        return base + " want me to suggest similar or popular alternatives?"
    if any(k in text for k in ["suggest", "recommend", "what should i eat", "what to eat"]):
        pop = _safe(get_popular, 5) or []
        rated = _safe(get_highest_rated, 5) or []
        merged = []
        seen = set()
        for row in pop + rated:
            n = row.get("item_name")
            if n and n not in seen:
                seen.add(n)
                merged.append(row)
        if not merged:
            merged = menu[:5]
        header = "here are a few great options:\n"
        return header + _format_list(merged, limit=8, with_rating=True)
    if mood_hint:
        return mood_hint
    return "i can show the menu, popular items, highest-rated dishes, budget-friendly picks, or category suggestions. try: 'show menu', 'popular items', 'highest rated', 'suggest category lunch', or 'suggest under ₹120'."

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply = _rule_engine(request.new_message)
    updated = request.history + [
        Content(role="user", parts=[Part(text=request.new_message)]),
        Content(role="model", parts=[Part(text=reply)]),
    ]
    return ChatResponse(reply=reply, updated_history=updated)

@router.get("/")
def ping():
    return {"ok": True, "hint": "POST /chat/chat with {history, new_message}"}
