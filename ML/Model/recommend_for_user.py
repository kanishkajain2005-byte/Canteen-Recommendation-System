import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\HP\Downloads\Canteen Recommendation System\ML\Model\trained_model.pkl"

def recommend_for_user(user_id, top_n=5):
    data = joblib.load(MODEL_PATH)
    pivot = data["pivot"]
    item_sim_df = data["item_sim"]

    if user_id not in pivot.index:
        print("⚠️ User not found. Returning most popular items instead.")
        popular = pivot.sum().sort_values(ascending=False).head(top_n)
        return list(popular.index)

    user_ratings = pivot.loc[user_id]
    user_rated_items = user_ratings[user_ratings > 0].index

    scores = {}
    for item in user_rated_items:
        similar_items = item_sim_df[item].sort_values(ascending=False)
        for sim_item, score in similar_items.items():
            if sim_item not in user_rated_items:
                scores[sim_item] = scores.get(sim_item, 0) + score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item for item, _ in sorted_scores[:top_n]]

    return recommendations

if __name__ == "__main__":
    user_id = "U101"  # example user from mock dataset
    recs = recommend_for_user(user_id)
    print(f"🍽️ Recommended items for {user_id}: {recs}")
