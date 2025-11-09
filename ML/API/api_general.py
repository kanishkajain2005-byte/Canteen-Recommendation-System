from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
import traceback
from fastapi.middleware.cors import CORSMiddleware

# ---- Import your recommender ----
# This assumes your repo structure is:
#   API/api_general.py
#   ML/Model/general_recommendation.py  (class: ContentBasedRecommender)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ML")))
from ML.Model.general_recommendation import ContentBasedRecommender  # noqa: E402

app = FastAPI(title="Canteen General Recommendation API")

# CORS for local dev + Render preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths ----------
# Repo root = one level up from /API
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Your main dataset for the recommender
DATA_PATH = os.path.join(BASE_DIR, "Data", "raw", "canteen_recommendation_dataset.csv")

# Precomputed model file
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "ML", "Model", "item_similarity.pkl"))

# ---------- Sanity checks ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Initialize recommender
recommender = ContentBasedRecommender(DATA_PATH)

try:
    if os.path.exists(MODEL_PATH):
        recommender.load_model(MODEL_PATH)
        print("✅ Loaded existing similarity model.")
    else:
        print("⚠️ No pre-trained model found. Building new one...")
        recommender.build_similarity_matrix()
        # Ensure model directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        recommender.save_model(MODEL_PATH)
except Exception:
    print("\n\n🔥 ERROR while preparing model 🔥")
    traceback.print_exc()
    print("🔥 END 🔥\n\n")
    # Don't crash startup—API can still respond with 500 for model endpoints

# ---------- Schemas ----------
class ItemRequest(BaseModel):
    item_name: str
    n: int = 5

# ---------- Health ----------
@app.get("/")
def root():
    return {"message": "Canteen General Recommendation API is running!"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

# ---------- New /menu endpoint ----------
@app.get("/menu")
def get_menu():
    """
    Reads Data/raw/menu.csv from the repo root.
    Make sure the file is committed to the repo so Render can read it.
    """
    try:
        menu_path = os.path.join(BASE_DIR, "Data", "raw", "menu.csv")
        if not os.path.exists(menu_path):
            raise FileNotFoundError(f"menu.csv not found at {menu_path}")

        df = pd.read_csv(menu_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/menu error: {e}")

# ---------- Recommendation endpoints ----------
@app.get("/recommend/popular")
def get_popular_items(limit: int = 10):
    try:
        popular_items = recommender.get_popular_items(n=limit)
        return popular_items.to_dict(orient="records")
    except Exception as e:
        print("\n\n🔥 ERROR TRACEBACK 🔥")
        traceback.print_exc()
        print("🔥 END TRACEBACK 🔥\n\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/similar")
def get_similar_items(item_name: str, limit: int = 5):
    try:
        normalized_name = item_name.strip().lower()
        similar_items = recommender.recommend_items(item_name=normalized_name, n=limit)
        return similar_items.to_dict(orient="records")
    except Exception as e:
        print(f"❌ Error in similar items: {e}")
        raise HTTPException(status_code=500, detail=str(e))
