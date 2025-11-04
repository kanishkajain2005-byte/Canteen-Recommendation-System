import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = r"C:\Users\HP\Downloads\Canteen Recommendation System\ML\Data\raw\canteen_orders.csv"
MODEL_PATH = r"C:\Users\HP\Downloads\Canteen Recommendation System\ML\Model\trained_model.pkl"

def retrain_model():
    if not os.path.exists(DATA_PATH):
        print("❌ Real dataset not found. Skipping retrain.")
        return

    print("📂 Loading new dataset...")
    df = pd.read_csv(DATA_PATH)

    # Ensure column names are correct
    if not {"user_id", "item_name", "quantity"}.issubset(df.columns):
        print("⚠️ Dataset missing required columns (user_id, item_name, quantity)")
        return

    print("🔧 Building user-item matrix...")
    pivot = df.pivot_table(index="user_id", columns="item_name", values="quantity", aggfunc="sum", fill_value=0)

    print("🧮 Computing cosine similarity...")
    item_sim = pd.DataFrame(
        cosine_similarity(pivot.T),
        index=pivot.columns,
        columns=pivot.columns
    )

    print("💾 Saving new model...")
    joblib.dump({"pivot": pivot, "item_sim": item_sim}, MODEL_PATH)
    print("✅ Model retrained and saved successfully!")

if __name__ == "__main__":
    retrain_model()
