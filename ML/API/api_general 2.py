from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
import traceback
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ML")))
from ML.Model.general_recommendation import ContentBasedRecommender  # noqa: E402

app = FastAPI(title="Canteen General Recommendation API (v2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "raw", "canteen_recommendation_dataset.csv")
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "ML", "Model", "item_similarity.pkl"))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

recommender = ContentBasedRecommender(DATA_PATH)

try:
    if os.path.exists(MODEL_PATH):
        recommender.load_model(MODEL_PATH)
        print("✅ Loaded existing similarity model.")
    else:
        print("⚠️ No pre-trained model found. Building new one...")
        recommender.build_similarity_matrix()
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        recommender.save_model(MODEL_PATH)
except Exception:
    print("\n\n🔥 ERROR while preparing model (v2) 🔥")
    traceback.print_exc()
    print("🔥 END 🔥\n\n")

class ItemRequest(BaseModel):
    item_name: str
    n: int = 5

@app.get("/")
def root():
    return {"message": "Canteen General Recommendation API (v2) is running!"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/menu")
def get_menu():
    try:
        menu_path = os.path.join(BASE_DIR, "Data", "raw", "menu.csv")
        if not os.path.exists(menu_path):
            raise FileNotFoundError(f"menu.csv not found at {menu_path}")

        df = pd.read_csv(menu_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/menu error: {e}")

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
