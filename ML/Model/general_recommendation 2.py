

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class ContentBasedRecommender:
    def __init__(self, data_path="data/canteen_recommendation_dataset.csv"):
       
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.similarity_df = None

        if "item_name" in self.df.columns:
            self.df["item_name"] = (
                self.df["item_name"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    def preprocess_data(self):
        
        df = self.df.copy()

        df['item_name'] = df['item_name'].astype(str).str.strip().str.lower()
        le = LabelEncoder()
        df['category_enc'] = le.fit_transform(df['category'])
        df['spicy_enc'] = le.fit_transform(df['spicy_level'])

        
        features = df[['item_id', 'category_enc', 'price', 'calories', 'spicy_enc', 'popularity_score']]
        features = features.drop_duplicates(subset='item_id').set_index('item_id')

        
        scaler = MinMaxScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        self.df = df 
        self.features_scaled = features_scaled
        return features_scaled

    def build_similarity_matrix(self):
        
        features_scaled = self.preprocess_data()
        similarity_matrix = cosine_similarity(features_scaled)
        self.similarity_df = pd.DataFrame(similarity_matrix, index=features_scaled.index, columns=features_scaled.index)
        return self.similarity_df

    def recommend_items(self, item_name, n=5):
    
        if self.similarity_df is None:
            self.build_similarity_matrix()

        
        self.df['item_name'] = self.df['item_name'].astype(str)

   
        item_name = str(item_name).strip().lower()

    
        matched_rows = self.df[self.df['item_name'].str.lower() == item_name]
        if matched_rows.empty:
            raise ValueError(f"Item '{item_name}' not found in dataset.")

        item_id = matched_rows['item_id'].values[0]

    
        similar_items = self.similarity_df[item_id].sort_values(ascending=False)[1:n+1]
        recommended_ids = similar_items.index

   
        recommendations = (
        self.df[self.df['item_id'].isin(recommended_ids)][['item_name', 'category', 'price']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

        return recommendations


    def get_popular_items(self, n=10):
    
        popular = (
        self.df.groupby("item_id")
        .agg({"purchase_count": "sum", "popularity_score": "mean"})
        .sort_values(by=["purchase_count", "popularity_score"], ascending=False)
        .head(n)
        .reset_index()
    )

    
        df_raw = pd.read_csv("/Users/Jaishreenirmala/Desktop/Canteen-Recommendation-System/ML/Data/raw/canteen_recommendation_dataset.csv")

    
        item_info = df_raw[["item_id", "item_name", "category", "price"]].drop_duplicates()
        popular = popular.merge(item_info, on="item_id", how="left")

    
        popular = popular.fillna({
        "item_name": "Unknown Item",
        "category": "Unknown",
        "price": 0,
        "popularity_score": 0,
        "purchase_count": 0
    })

    
        popular["popularity_score"] = popular["popularity_score"].round(2)
        popular["price"] = popular["price"].round(2)

    
        return popular[["item_name", "purchase_count", "popularity_score", "category", "price"]]





    def save_model(self, path='Model/item_similarity.pkl'):
        
        os.makedirs("Model", exist_ok=True)
        if self.similarity_df is None:
            self.build_similarity_matrix()
        with open(path, 'wb') as f:
            pickle.dump(self.similarity_df, f)

    def load_model(self, path='Model/item_similarity.pkl'):
        
        with open(path, 'rb') as f:
            self.similarity_df = pickle.load(f)
