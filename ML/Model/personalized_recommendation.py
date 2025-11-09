import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from bson import ObjectId

class PersonalizedRecommender:
    def __init__(self, mongo_client, db_name="auth-db"):
        self.mongo_client = mongo_client
        self.db = self.mongo_client[db_name]
        self.collection = self.db["purchases"]
        self.user_item_matrix = None
        self.similarity_df = None

    async def fetch_data(self):
        
        cursor = self.collection.find({})
        records = []

        async for doc in cursor:  
            user_id = str(doc.get("userId"))
            for item in doc.get("items", []):
                item_id = str(item.get("itemId", "unknown"))
                amount = item.get("totalAmount", 1)
                records.append({
                    "userId": user_id,
                    "itemId": item_id,
                    "amount": amount
                })

        if not records:
            raise ValueError("❌ No purchase data found in MongoDB.")

        df = pd.DataFrame(records)
        print(f"✅ Loaded {len(df)} purchase records from MongoDB")
        return df

    async def build_user_item_matrix(self):
        """Build the user-item matrix asynchronously."""
        df = await self.fetch_data()
        self.user_item_matrix = df.pivot_table(
            index="userId",
            columns="itemId",
            values="amount",
            fill_value=0
        )
        print(f"✅ Created user-item matrix with shape {self.user_item_matrix.shape}")
        return self.user_item_matrix

    async def train_model(self):
        """Train the personalized user similarity model."""
        if self.user_item_matrix is None:
            await self.build_user_item_matrix()

        similarity = cosine_similarity(self.user_item_matrix)
        self.similarity_df = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        print("✅ Personalized model (user-user similarity) built.")
        return self.similarity_df

    def save_model(self, path="ML/Model/personalized_model.pkl"):
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.similarity_df, f)
        print(f"✅ Personalized model saved at: {path}")

    def load_model(self, path="ML/Model/personalized_model.pkl"):
        """Load the saved similarity model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model not found at {path}")
        with open(path, "rb") as f:
            self.similarity_df = pickle.load(f)
        print(f"✅ Personalized model loaded from: {path}")

    def recommend_for_user(self, user_id, n=5):
        
        if self.similarity_df is None:
            raise ValueError("Model not trained or loaded.")

        if user_id not in self.similarity_df.index:
            raise ValueError(f"User {user_id} not found in similarity matrix.")

        
        similar_users = self.similarity_df[user_id].sort_values(ascending=False)[1:n+1]
        top_users = similar_users.index.tolist()

        
        user_purchases = self.user_item_matrix.loc[user_id]
        already_bought = set(user_purchases[user_purchases > 0].index)

        rec_scores = self.user_item_matrix.loc[top_users].mean().sort_values(ascending=False)
        rec_items = [item for item in rec_scores.index if item not in already_bought][:n]

        print(f"🎯 Recommended items for {user_id}: {rec_items}")
        return rec_items
