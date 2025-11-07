# predict_chatbot.py
import pandas as pd
import joblib
from pathlib import Path

# --- Configuration (Must match your training file paths) ---

# Get the path to the directory containing the 'Model' folder
# NOTE: This assumes this script is run from the project root or the same level as the 'ML' directory
ML_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'ML' else Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "Model" / "trained_model.pkl"

# --- Core Recommendation Function ---

def recommend_items(user_id: str, pivot_matrix: pd.DataFrame, item_similarity: pd.DataFrame, n_recommendations: int = 5) -> list[str]:
    """
    Generates a list of recommended items for a given user based on item similarity.
    
    Args:
        user_id: The ID of the user to get recommendations for.
        pivot_matrix: The user-item quantity matrix (from trained_model.pkl).
        item_similarity: The item-to-item cosine similarity matrix (from trained_model.pkl).
        n_recommendations: The number of items to recommend.
        
    Returns:
        A list of recommended item names.
    """
    # 1. Get the user's order history vector from the pivot matrix
    try:
        user_vector = pivot_matrix.loc[user_id]
    except KeyError:
        return [f"Error: User ID '{user_id}' not found in the training data."]

    # 2. Get the list of items the user has already ordered (to filter later)
    ordered_items = user_vector[user_vector > 0].index.tolist()
    
    # 3. Calculate the total recommendation score for all items
    recommendation_scores = {}
    
    # Iterate through every item the user HAS ordered
    for ordered_item in ordered_items:
        # Get the similarity scores for this ordered item against ALL other items
        similar_items = item_similarity.loc[ordered_item]
        
        # Multiply the similarity scores by the user's quantity/rating for the ordered item
        # and add it to the total recommendation score
        quantity = user_vector[ordered_item]
        weighted_scores = similar_items * quantity
        
        for item, score in weighted_scores.items():
            if item not in ordered_items: # Do not recommend items they've already bought
                recommendation_scores[item] = recommendation_scores.get(item, 0) + score

    # 4. Sort the scores and get the top N recommendations
    recommended_items = sorted(
        recommendation_scores.items(), 
        key=lambda item: item[1], 
        reverse=True
    )

    # 5. Extract just the item names
    return [item[0] for item in recommended_items[:n_recommendations]]


# --- Chatbot Simulation / Main Execution ---

if __name__ == "__main__":
    print("🤖 Starting Canteen Recommender Chatbot...")
    
    # Load the trained model assets
    try:
        loaded_model = joblib.load(MODEL_PATH)
        PIVOT = loaded_model["pivot"]
        ITEM_SIM = loaded_model["item_sim"]
        print(f"✅ Model loaded from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}. Run 'train_recommender.py' first.")
        exit()
    except KeyError:
        print("❌ ERROR: Model file content is incorrect. Expected 'pivot' and 'item_sim' keys.")
        exit()

    # Chatbot Loop
    while True:
        # Prompt user for input
        user_input = input("\nChatbot: Enter User ID (e.g., 'U1', 'U2') to get recommendations or type 'exit': ").strip()
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        if not user_input:
            continue

        # Get recommendations using the loaded assets
        recommendations = recommend_items(user_input, PIVOT, ITEM_SIM)
        
        # Display the result to the user (the chatbot's response)
        if recommendations and recommendations[0].startswith("Error"):
            print(f"Chatbot: {recommendations[0]}")
        else:
            print(f"Chatbot: For User '{user_input}', I recommend the following items:")
            for i, item in enumerate(recommendations, 1):
                print(f"  {i}. {item}")
        
    print("\n")