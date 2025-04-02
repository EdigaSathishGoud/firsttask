import streamlit as st
import pandas as pd
import io

st.title("Streamlit UI with Multiple Messages and File Upload")

# File upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Initialize session state variables if not already set
if "processed" not in st.session_state:
    st.session_state["processed"] = False
    st.session_state["file_name"] = None

if uploaded_file:
    st.write("✅ File uploaded successfully:", uploaded_file.name)
    
    # Process button
    if st.button("Process"):
        st.session_state["processed"] = True
        st.session_state["file_name"] = uploaded_file.name

# If processing has been done, display messages
if st.session_state["processed"]:
    st.write("🔹 Message 1: Welcome to the App!")
    st.write("🔹 Message 2: Your uploaded file is:", st.session_state["file_name"])
    st.write("🔹 Message 3: Processing is complete.")
    st.write("🔹 Message 4: AI is amazing!")
    st.write("🔹 Message 5: Hope you're having a great day! 😊")

    # Create downloadable content
    def create_download():
        output = io.BytesIO()
        output.write("Processing complete!".encode())  # Custom message for download
        output.seek(0)
        return output

    st.download_button(
        label="Download Message",
        data=create_download(),
        file_name="message.txt",
        mime="text/plain"
    )
