import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path # 


BASE_DIR = Path(r"C:\Users\HP\Downloads\Canteen Recommendation System\ML")
DATA_PATH = BASE_DIR / "Data" / "raw" / "canteen_orders.csv"
MODEL_PATH = BASE_DIR / "Model" / "trained_model.pkl"
REQUIRED_COLS = {"user_id", "item_name", "quantity"}

def retrain_model():
    """
    Loads raw canteen order data, computes a user-item pivot table,
    calculates item-to-item cosine similarity, and saves the pivot
    table and similarity matrix as a model file.
    """
    
    if not DATA_PATH.exists():
        print(f"❌ Real dataset not found at: {DATA_PATH}. Skipping retrain.")
        return

    print("📂 Loading new dataset...")
    df = pd.read_csv(DATA_PATH)

    # 2. Robust column name check
    if not REQUIRED_COLS.issubset(set(df.columns)):
        print(f"⚠️ Dataset missing required columns ({', '.join(REQUIRED_COLS)})")
        return

    print("🔧 Building user-item matrix...")
    # Use explicit fill_value=0 for clarity in the pivot table
    pivot = df.pivot_table(
        index="user_id", 
        columns="item_name", 
        values="quantity", 
        aggfunc="sum", 
        fill_value=0
    )

    print("🧮 Computing cosine similarity...")
    # Transpose the pivot table (pivot.T) for item-item similarity
    item_sim = pd.DataFrame(
        cosine_similarity(pivot.T),
        index=pivot.columns,
        columns=pivot.columns
    )

    print("💾 Saving new model...")
    # 3. Ensure the target directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the necessary components
    joblib.dump({"pivot": pivot, "item_sim": item_sim}, MODEL_PATH)
    print(f"✅ Model retrained and saved successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    retrain_model()