from fastapi import FastAPI
import pandas as pd
from datetime import timedelta

app = FastAPI(title="Canteen Recommendation API")

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\Canteen Recommendation System\ML\Data\raw\mock_canteen_orders.csv", parse_dates=["timestamp"])

# --- Recommender Logic ---
def get_recommendations(df, mode="popular", top_n=5):
    if mode == "popular":
        data = (
            df.groupby("item_name")["quantity"]
              .sum()
              .reset_index()
              .sort_values(by="quantity", ascending=False)
        )
    elif mode == "trending":
        recent_period = df["timestamp"].max() - timedelta(days=7)
        data = (
            df[df["timestamp"] >= recent_period]
              .groupby("item_name")["quantity"]
              .sum()
              .reset_index()
              .sort_values(by="quantity", ascending=False)
        )
    else:
        raise ValueError("Mode must be 'popular' or 'trending'")
    return data.head(top_n)["item_name"].tolist()

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Welcome to the Canteen Recommendation API"}

@app.get("/recommend")
def recommend(mode: str = "popular", top_n: int = 5):
    try:
        recommendations = get_recommendations(df, mode, top_n)
        return {"mode": mode, "recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}
