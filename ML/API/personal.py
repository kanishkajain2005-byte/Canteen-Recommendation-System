from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import motor.motor_asyncio
import os
import sys
from dotenv import load_dotenv
from ML.Model.personalized_recommendation import PersonalizedRecommender

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
print("✅ Added to sys.path:", ROOT_DIR)

load_dotenv()

app = FastAPI(title="Personalized Recommendation API")

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

recommender = PersonalizedRecommender(mongo_client)

@app.on_event("startup")
async def startup_event():
    try:
        await mongo_client.admin.command('ping')
        print("✅ MongoDB connected successfully!")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)

class UserRequest(BaseModel):
    user_id: str
    top_n: int = 5

@app.post("/train")
async def train_model():
    """Train and save the personalized model."""
    try:
        await recommender.train_model()   # ✅ await here
        recommender.save_model()
        return {"message": "✅ Personalized model trained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
async def recommend_items(request: UserRequest):
    """Fetch personalized recommendations for a user."""
    try:
        recommender.load_model()
        rec_items = recommender.recommend_for_user(request.user_id, n=request.top_n)
        return {"user_id": request.user_id, "recommended_items": rec_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Personalized Recommendation API is running 🚀"}
