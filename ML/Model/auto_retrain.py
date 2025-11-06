from pathlib import Path
import pandas as pd, joblib
from sklearn.metrics.pairwise import cosine_similarity

ML_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ML_DIR / "Data" / "raw" / "mock_canteen_orders.csv"
MODEL_PATH = ML_DIR / "Model" / "trained_model.pkl"

def retrain():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_name"] = df["item_name"].astype(str)

    pivot = df.pivot_table(index="user_id", columns="item_name", values="quantity", aggfunc="sum", fill_value=0)
    sim = cosine_similarity(pivot.T)
    item_sim = pd.DataFrame(sim, index=pivot.columns, columns=pivot.columns)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pivot": pivot, "item_sim": item_sim}, MODEL_PATH)

    print("✅ Retrained model saved at:", MODEL_PATH)

if __name__ == "__main__":
    retrain()
