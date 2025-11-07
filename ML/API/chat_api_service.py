import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
import uvicorn

# Load environment variables (GEMINI_API_KEY)
load_dotenv() 

class Part(BaseModel):
    text: str

# Define the structure of a single message turn (user or model)
class Content(BaseModel):
    role: str = Field(..., pattern="^(user|model)$") # Role must be 'user' or 'model'
    parts: List[Part]

# Define the structure of the incoming request
class ChatRequest(BaseModel):
    """Request body containing the full conversation history and the new message."""
    
    # List of past messages (user/model turns)
    history: List[Content] 
    
    # The new message from the user
    new_message: str

# Define the structure of the outgoing response
class ChatResponse(BaseModel):
    """Response body containing the model's reply and the updated full history."""
    
    reply: str
    updated_history: List[Content]

# --- FastAPI Setup and Client Initialization ---

app = FastAPI(
    title="Gemini Conversational API",
    description="A service for multi-turn chat using the Gemini API and persistent history."
)

# Initialize Gemini Client globally
try:
    gemini_client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    gemini_client = None
    print(f"CRITICAL: Failed to initialize Gemini Client. Check API Key. Error: {e}")


# --- API Endpoint: /chat ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives chat history, gets a response from Gemini, and returns the updated history.
    """
    if not gemini_client:
        raise HTTPException(
            status_code=503, 
            detail="AI service is unavailable. Check server configuration."
        )

    conversation_contents = []
    
    # Load existing history
    for message in request.history:
        conversation_contents.append(message.model_dump())
    
    # Append the new user message
    new_user_content = {
        "role": "user",
        "parts": [{"text": request.new_message}]
    }
    conversation_contents.append(new_user_content)

    try:
        # 2. Call the Gemini API with the full history
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation_contents
        )
        
        # Extract the model's reply
        model_reply = response.text
        
        # 3. Create the new model Content object and update history
        new_model_content = Content(
            role="model",
            parts=[Part(text=model_reply)]
        )
        
        # Prepare the response payload: current history + new user message + new model reply
        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            new_model_content
        ]
        
        # 4. Return the response
        return ChatResponse(
            reply=model_reply,
            updated_history=updated_history
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Gemini API error: {e}"
        )

# --- Optional: Test Endpoint ---
@app.get("/")
def home():
    return {"message": "Conversational Chat API is running. Go to /docs to test the /chat endpoint."}


if __name__ == "__main__":
    
    uvicorn.run("chat_api_service:app", host="127.0.0.1", port=8001, reload=True)
