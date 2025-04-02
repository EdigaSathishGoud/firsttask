import streamlit as st
import json
from openai import AzureOpenAI

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to load saved conversations from file
def load_conversations():
    try:
        with open("conversations.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Function to save conversations to file
def save_conversations(conversations):
    with open("conversations.json", "w") as file:
        json.dump(conversations, file)

# Load saved conversations
conversations = load_conversations()

def llm_response(query):
    client = AzureOpenAI(
        api_key="836fea48dd7a402dbd8cf73f4dddb2e5",
        api_version="2023-05-15",
        azure_endpoint="https://bayerbichatgpt4.openai.azure.com/"
    )

    response = client.chat.completions.create(
        model="BayerBIGPT35Turbo",    
        messages=[        
            {"role": "user", "content": f"You are an expert in answering user queries. \n A query to answer: {query}"}
        ],
        temperature=0,
        max_tokens=1024
    )

    return response.choices[0].message.content

st.header("Chat History")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter text to answer"):
    # Check if prompt has been asked before
    if prompt in conversations:
        st.info("Fetched from History")
        response = conversations[prompt]
    else:
        st.info("LLM response")
        # Generate response using LLM
        response = llm_response(prompt)
        # Save the new conversation
        conversations[prompt] = response
        save_conversations(conversations)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display assistant response in chat message container
    st.chat_message("assistant").markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
