from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import timedelta

app = FastAPI(title="Canteen Recommendation API", version="1.1.0")

# CORS — allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Repo-relative paths
BASE_DIR = Path(__file__).resolve().parent      # ML/API
ML_DIR = BASE_DIR.parent                        # ML
DATA_PATH = (ML_DIR / "Data" / "raw" / "mock_canteen_orders.csv").resolve()
MODEL_PATH = (ML_DIR / "Model" / "trained_model.pkl").resolve()


# Load orders CSV
def load_orders() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])   # drop invalid timestamps
    return df

def get_popular(df: pd.DataFrame, top_n: int = 5, days: Optional[int] = None):
    if days and "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        
        # ✅ Create timezone-naive cutoff
        cutoff = (pd.Timestamp.utcnow() - timedelta(days=days)).tz_localize(None)

        # ✅ Ensure timestamps are valid & tz-naive
        df = df[df["timestamp"].notna()]
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # ✅ Safe comparison
        df = df[df["timestamp"] >= cutoff]

    counts = df["item_name"].value_counts().reset_index()
    counts.columns = ["item_name", "order_count"]

    return counts.head(top_n).to_dict(orient="records")






# Popularity-based recommender
def get_popular(df: pd.DataFrame, top_n: int = 5, days: Optional[int] = None):
    if days and "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):

        # ✅ make sure timestamps are timezone-naive
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # ✅ cutoff must be timezone-naive too
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(days=days)

        # ✅ safe filtering
        df = df[df["timestamp"] >= cutoff]

    counts = df["item_name"].value_counts().reset_index()
    counts.columns = ["item_name", "order_count"]

    return counts.head(top_n).to_dict(orient="records")




# Try personalized (Phase 2)
def try_personalized(user_id: str, top_n: int = 5):
    try:
        import joblib
        data = joblib.load(MODEL_PATH)

        pivot = data["pivot"]
        item_sim = data["item_sim"]

        if user_id not in pivot.index:
            return None  # cold start fallback

        user_vec = pivot.loc[user_id]
        scores = (item_sim * user_vec).sum(axis=1).sort_values(ascending=False)

        purchased = set(pivot.columns[user_vec > 0])
        recs = [i for i in scores.index if i not in purchased][:top_n]

        return [{"item_name": item, "source": "personalized"} for item in recs]

    except Exception:
        return None


# ROUTES ----------------------------------------------------------

@app.get("/")
def health():
    return {"ok": True, "service": "canteen-recommendation-api"}


@app.get("/recommend")
def recommend(
    top_n: int = Query(5, ge=1, le=50),
    window_days: Optional[int] = Query(None, ge=1),
):
    df = load_orders()
    recs = get_popular(df, top_n=top_n, days=window_days)
    return {
        "mode": "popular",
        "top_n": top_n,
        "window_days": window_days,
        "recommendations": recs,
    }


@app.get("/recommend/user/{user_id}")
def recommend_user(
    user_id: str,
    top_n: int = Query(5, ge=1, le=50),
):
    # Try personalized
    recs = try_personalized(user_id, top_n=top_n)
    if recs:
        return {
            "mode": "personalized",
            "user_id": user_id,
            "top_n": top_n,
            "recommendations": recs,
        }

    # Fallback
    df = load_orders()
    fallback = get_popular(df, top_n=top_n)
    return {
        "mode": "popular-fallback",
        "user_id": user_id,
        "top_n": top_n,
        "recommendations": fallback,
    }
