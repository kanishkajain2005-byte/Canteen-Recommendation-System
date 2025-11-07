# main.py
from fastapi import FastAPI
from models import RecommendationRequest, RecommendationResponse
from ml_logic.recommender import recommend_items, load_recommender_assets
import uvicorn

# Initialize the FastAPI application
app = FastAPI(
    title="Canteen Item Recommender API",
    description="A collaborative filtering service for canteen item recommendations."
)

# --- Lifespan Event: Load model when the application starts ---
@app.on_event("startup")
def startup_event():
    """Loads the ML assets when the FastAPI server starts."""
    load_recommender_assets()

# --- API Endpoints ---

@app.get("/")
def home():
    """Health check endpoint."""
    return {"status": "running", "service": "Canteen Recommender API"}

@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get Item Recommendations for a User"
)
async def get_recommendations(request: RecommendationRequest):
    """
    Takes a User ID and returns a list of recommended canteen items 
    based on collaborative filtering.
    """
    user_id = request.user_id
    
    # Call the recommendation logic (the prediction step)
    recommendations_result = recommend_items(user_id)
    
    # Check if the result is an error message (str) or a list of items (list[str])
    if isinstance(recommendations_result, str):
        # Handle known errors (e.g., User not found, Model not loaded)
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[],
            status=f"error: {recommendations_result}"
        )
    
    # Success case: returns the list of recommendations
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations_result,
        status="success"
    )

if __name__ == "__main__":
    # Command to run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)