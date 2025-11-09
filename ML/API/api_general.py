# API/general_recommendation_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys, os
import pandas as pd
import traceback

# Add path to access ML/Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ML/Model")))

from Model.general_recommendation import ContentBasedRecommender


# Initialize FastAPI app
app = FastAPI(title="Canteen General Recommendation API")

# Path to dataset and model
DATA_PATH = os.path.abspath("../ML/Data/raw/canteen_recommendation_dataset.csv")
MODEL_PATH = os.path.abspath("../ML/Model/item_similarity.pkl")

# Initialize recommender
recommender = ContentBasedRecommender(DATA_PATH)

# Load pre-trained similarity matrix if available
try:
    recommender.load_model(MODEL_PATH)
    print("✅ Loaded existing similarity model.")
except:
    print("⚠️ No pre-trained model found. Building new one...")
    recommender.build_similarity_matrix()
    recommender.save_model(MODEL_PATH)

# Request model for “similar items”
class ItemRequest(BaseModel):
    item_name: str
    n: int = 5

@app.get("/")
def root():
    return {"message": "Canteen General Recommendation API is running!"}

@app.get("/recommend/popular")
def get_popular_items(limit: int = 10):
    
    try:
        popular_items = recommender.get_popular_items(n=limit)
        return popular_items.to_dict(orient="records")
    except Exception as e:
        print("\n\n🔥 ERROR TRACEBACK 🔥")
        traceback.print_exc()  # 👈 This will show the *exact* error line and file in terminal
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

