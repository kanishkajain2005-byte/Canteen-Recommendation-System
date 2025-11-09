# ML/Model/train_personalized_model.py

import os
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from dotenv import load_dotenv
from personalized_recommendation import PersonalizedHybridRecommender


load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGODB_DB", "canteen")
MODEL_PATH = "ML/Model/personalized_model.pkl"

async def train_model():
    """Fetch users, items, and orders from MongoDB, then train and save model."""
    print("📡 Connecting to MongoDB...")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]

    users = await db.users.find().to_list(length=None)
    items = await db.items.find().to_list(length=None)
    orders = await db.orders.find().to_list(length=None)

    if not users or not orders or not items:
        raise ValueError("❌ Missing data in MongoDB collections (users, items, or orders).")

    print(f"✅ Loaded: {len(users)} users, {len(items)} items, {len(orders)} orders")

    # Flatten orders into (user_id, item_id, rating)
    order_records = []
    for order in orders:
        user_id = str(order.get("userId"))
        for item in order.get("items", []):
            item_id = str(item.get("item_id") or item.get("_id"))
            if item_id:
                order_records.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 1
                })

    users_df = pd.DataFrame(users)
    items_df = pd.DataFrame(items)
    orders_df = pd.DataFrame(order_records)

    if users_df.empty or orders_df.empty:
        raise ValueError("⚠️ Not enough data to train the model.")

    print("🧠 Training the personalized hybrid model...")
    recommender = PersonalizedHybridRecommender(model_path=MODEL_PATH)
    recommender.train(users_df, items_df, orders_df)
    recommender.save()
    print(f"✅ Model trained and saved at {MODEL_PATH}")

    client.close()

if __name__ == "__main__":
    asyncio.run(train_model())
