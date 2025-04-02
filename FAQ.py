import streamlit as st

# Define FAQ data
faq_data = {
    "What is this app about?": "This app helps you...",
    "How do I get started?": "To get started, simply...",
    "Can I customize settings?": "Yes, you can customize...",
    # Add more questions and answers as needed
}

# Main Streamlit app
st.title("FAQ App")

# Input field for user query
user_question = st.text_input("Enter your question:")

# Display FAQs when input field is focused
if user_question == "":
    st.write("Frequently Asked Questions:")
    for question in faq_data:
        st.write(f"- {question}")
