import streamlit as st
import requests

# 1. Configuration and API Endpoint
# Ensure your FastAPI backend is running on this address/port
API_URL = "http://localhost:8001/chat"

st.set_page_config(page_title="Minimal Chatbot Frontend", layout="centered")
st.title("🤖 Chatbot Interface")

# 2. Initialize Session State for Chat History
# Streamlit uses st.session_state to store data across user interactions
if "messages" not in st.session_state:
    # Start with an empty list for the conversation
    st.session_state.messages = []

# 3. Display Chat History
# Loop through the history and display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if prompt := st.chat_input("Ask a question..."):
    # Add user's message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare message history for the API call (FastAPI expects a list of messages)
    history_for_api = st.session_state.messages

    # Display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Make the POST request to the FastAPI backend
                response = requests.post(API_URL, json={"history": history_for_api})
                
                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data.get("response", "Error: No response content from API.")
                else:
                    # Handle API errors
                    assistant_response = f"Error: API returned status code {response.status_code}. Is the backend running on port 8001?"
            
            except requests.exceptions.ConnectionError:
                # Handle connection failure (most common issue: backend is not running)
                assistant_response = "Error: Could not connect to the API server. Please ensure your **FastAPI backend** is running on `http://localhost:8001`."
            
            except Exception as e:
                assistant_response = f"An unexpected error occurred: {e}"

        # Display the final response
        st.markdown(assistant_response)

    # 5. Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})