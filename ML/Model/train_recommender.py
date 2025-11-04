import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

DATA_PATH = r"C:\Users\HP\Downloads\Canteen Recommendation System\ML\Data\raw\mock_canteen_orders.csv"
MODEL_PATH = r"C:\Users\HP\Downloads\Canteen Recommendation System\ML\Model\trained_model.pkl"

print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# Basic cleaning
df["user_id"] = df["user_id"].astype(str)
df["item_name"] = df["item_name"].astype(str)

# Create user–item matrix
pivot = df.pivot_table(index="user_id", columns="item_name", values="quantity", aggfunc="sum", fill_value=0)

# Compute cosine similarity between items
print("🔍 Computing item similarity matrix...")
similarity_matrix = cosine_similarity(pivot.T)
item_sim_df = pd.DataFrame(similarity_matrix, index=pivot.columns, columns=pivot.columns)

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({"pivot": pivot, "item_sim": item_sim_df}, MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")
