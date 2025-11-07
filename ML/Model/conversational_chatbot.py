import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Configuration ---

# Load environment variables (like GEMINI_API_KEY) from a .env file
load_dotenv() 

# --- Chatbot Core Functionality ---

def run_chatbot():
    """
    Initializes the Gemini client and runs a persistent, multi-turn chat session 
    in the terminal.
    """
    
    # 1. Initialize Gemini Client
    try:
        # Client automatically looks for GEMINI_API_KEY in environment variables
        client = genai.Client()
        MODEL_NAME = "gemini-2.5-flash" 
        print(f"🤖 Initializing Gemini Chatbot with model: {MODEL_NAME}...")
    except Exception as e:
        print("❌ ERROR: Failed to initialize Gemini Client.")
        print("Please ensure you have set your GEMINI_API_KEY in your .env file.")
        sys.exit(1)

    # 2. Create Chat Session
    # The start_chat method handles the history automatically
    # We provide a system instruction to set the chatbot's personality
    chat = client.chats.create(
        model=MODEL_NAME, 
        config=types.GenerateContentConfig(
            system_instruction="You are a friendly, witty, and helpful AI assistant named Chip. Keep your responses concise and engaging."
        )
    )

    print("\n------------------------------------------------------")
    print("👋 Hi! I'm Chip, your friendly AI assistant. Ask me anything!")
    print("   Type 'exit' or 'quit' to end the session.")
    print("------------------------------------------------------")

    # 3. Main Chat Loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nChip: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue

            # Send message and get the response
            # Using stream=True provides a better, typewriter-like user experience
            response_stream = chat.send_message(user_input, stream=True)
            
            print("Chip: ", end="")
            
            # Print the response in real-time as chunks arrive
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    print(chunk.text, end="")
                    full_response += chunk.text
            
            sys.stdout.flush() # Ensure all output is printed immediately
            print() # Newline after the full response is streamed

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nChip: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nChip: Oops! An error occurred: {e}. Please try again.")

# --- Execute Chatbot ---

if __name__ == "__main__":
    # Ensure environment is ready before running
    if not os.getenv("GEMINI_API_KEY"):
        print("CRITICAL: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
        
    run_chatbot()
