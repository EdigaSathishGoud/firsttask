import streamlit as st
import pandas as pd
st.title("POC")
# Take user input
user_input = st.text_input("Enter a string to proceed:")

# Check if input is a non-empty string
if user_input.strip():  
    # Define functions to get values
    def get_a():
        return 5

    def get_b():
        return 10

    def get_c():
        return 12

    # Fetch values from functions
    a = get_a()
    b = get_b()
    c = get_c()

    # Store data in a dictionary
    data = {"Variable": ["a", "b", "c"], "Value": [a, b, c]}

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Display in Streamlit
    st.title("Variable Values Table")
    st.table(df) 

