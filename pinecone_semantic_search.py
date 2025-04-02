import streamlit as st
import json
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import ast
import os
from pinecone import Pinecone

api_key = 'c4f6971b-50d6-4c70-ac25-6b02638c31f1'
# configure client
pc = Pinecone(api_key=api_key)

index_name = "semantic"

# connect to index
index = pc.Index(index_name)
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

st.header("Semantic Search")

#load the model
model1 = SentenceTransformer('all-mpnet-base-v2')
def main():
    
    if prompt := st.chat_input("Enter text to answer"): # Check if prompt has been asked before
        score=0
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        xq=model1.encode([prompt]).tolist()
        results = index.query(vector=xq,top_k=1,include_metadata=True,namespace="ns1")
        print(results)
    
        for result in results['matches']:
            id,score,ans,counter=result['id'],round(result['score'], 2),list(result['metadata'].values())[0],list(result['metadata'].values())[2]

        print("-------",id,counter)
        if score>0.8:
            st.write(f"Found in History with score {score}")
            response = ans
            index.update(id=id, set_metadata={"count": counter+1},namespace="ns1")
        else:
            st.write(f"Not Found in History,so generating response using LLM")
            # Generate response using LLM
            response = llm_response(prompt)
            # Save the new conversation
            index.upsert(vectors=[{'id':prompt, 'values': model1.encode(prompt).tolist(),'metadata': {'Question': prompt, 'Answer': response,'count': 1}}],namespace="ns1")
            st.info("Saved to History")
            

        # if prompt in conversations:
        #     st.info("Fetched from History")
        #     response = conversations[prompt]
        # else:
        #     st.info("LLM response")
        #     # Generate response using LLM
        #     response = llm_response(prompt)
        #     # Save the new conversation
        #     conversations[prompt] = response
        #     save_conversations(conversations)

        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        st.chat_message("assistant").markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if __name__ == "__main__":
    main()