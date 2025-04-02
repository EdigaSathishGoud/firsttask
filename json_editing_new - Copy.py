import pandas as pd
import re
import streamlit as st
import io
import json
import base64
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI


def create_nested_dict_from_excel(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Read the relevant sheet and skip the first few rows (adjust as necessary)
    df = pd.read_excel(xls, sheet_name='Sheet1') # , skiprows=4

    # Drop rows where all values are NaN
    df_cleaned = df.dropna(how='all')

    # Replace NaN values with empty strings
    # df_filled = df_cleaned.fillna("")

    # print(df)
    # exit()
    
    # Drop columns that are completely empty (all NaN values)
    # df = df.dropna(axis=1, how='all')
    
    # Initialize the nested dictionary
    nested_dict = {}

    # Iterate over the rows in the dataframe
    for _, row in df_cleaned.iterrows():
        derivation_logic = str(row['Derivation logic'])
        # filter = str(row['Filter'])
        
        # Clean all multi-line strings by removing line breaks and tabs
        # print(f"\nDerivation Logic {i}: ", repr(row
        cleaned_derivation_logic = re.sub(r'\s+', ' ', derivation_logic.strip())
        # cleaned_filter = re.sub(r'\s+', ' ', filter.strip())
        # print(f"\nCleaned Value {i}: ", repr(cleaned_value))

        Projection = row['Projection']

        inner_dict = {
            # 'Description': row['Description'],
            'Field Technical Name': row['Field Technical Name'],
            'Derivation Type': row['Derivation Type'],
            'Derivation logic': cleaned_derivation_logic,
            'Source': row['Source'],
            # 'Filter': cleaned_filter,
            'Excluded columns': row['Excluded columns'],
            'Natural Language Derivation': row['Natural Language Derivation'],
            'JOIN KEYS': row['JOIN KEYS']
        }

        if Projection in nested_dict:
            nested_dict[Projection].append(inner_dict)
        else:
            nested_dict[Projection] = [inner_dict]

    return nested_dict


def process_dictionary(data):
    # print("entered process_dictinary function")
    for key, values in data.items():
        # print("Key is :\n",key,"\nvalues are :\n",values)
        for entry in values:
            if entry.get("Derivation Type") == "filter":
                derivation_logic = entry.get("Derivation logic")
                # print(f"Derivation_logic for table ,{key} is:",derivation_logic)
                if derivation_logic and derivation_logic.lower() != "nan":
                    # print("calling llm:\n")
                    query = call_llm_for_query(derivation_logic)
                    query= query.replace("\n","")
                    # print("\nllm response: \n",query)
                    entry["Derivation logic"] = query
            if entry.get("Derivation Type") == "Derive":
                derivation_logic = entry.get("Derivation logic")
                if derivation_logic and derivation_logic.lower() != "nan":
                    query = call_llm_for_query(derivation_logic)
                    query= query.replace("\n","")
                    # Ensure consistent spacing in the query
                    query = re.sub(r'\s+', ' ', query).strip()
                    # Extracting the portion between SELECT and FROM
                    match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)

                    if match:
                        extracted_expression = match.group(1).strip()
                        # Remove alias (handles both "AS alias_name" and implicit alias without "AS")
                        extracted_expression = re.sub(r'\s+AS\s+\w+', '', extracted_expression, flags=re.IGNORECASE)
                        extracted_expression = re.sub(r'\s+\w+$', '', extracted_expression)  # Removes trailing alias if no "AS"
                        entry["Derivation logic"] = extracted_expression
                    else:
                        entry["Derivation logic"] = query
                    # print("\nllm response: \n",query)
                    


    return data

def call_llm_for_query(derivation_logic):
    """
    Function to call LLM and generate a query based on the derivation logic.
    Replace this with actual API calls to your LLM (e.g., Azure OpenAI).
    """
    client = AzureOpenAI(
    api_key = "c1ed1b7df1ad42b68fd5da1cef21491a",  
    api_version = "2023-05-15",
    azure_endpoint = "https://goldvmpoc.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2024-08-01-preview"
    )
    response = client.chat.completions.create(
    model="gpt-4-32k", 
        messages=[
            {"role": "system", "content": """You are an expert in writing conditions or operations in SQL queries.Follow below instructions
             
            Instructions:
             If Derivation logic is "Concatenation of a, b and c" then output should only contain CONCAT(a,b,c)
             Do not include Select , from , table names etc in the output . Provide only the condition which satisfies the given input.
             Dont include ```sql at the beginning and ``` at the end of the generated output
             """},
            {"role": "user", "content": f"Generate an SQL query condition for the following logic: {derivation_logic}"}
            
        ],temperature=0)
    return response.choices[0].message.content


def generate_json(excel_dict, asset_name, view_comment, common_column_names):
    client = AzureOpenAI(
    api_key = "c1ed1b7df1ad42b68fd5da1cef21491a",  
    api_version = "2023-05-15",
    azure_endpoint = "https://goldvmpoc.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2024-08-01-preview"
    )

    response = client.chat.completions.create(
    model="gpt-4-32k",    
    messages=[
        {"role": "user", "content": f"""
You are a SQL and JSON expert. I want you to generate JSON structure in the exact format as provided below. Inside the format, I've provided instructions on how to generate the values for respective keys. Refer the data provided in the dictionary according to the given instructions.

---

Sample structure of provided dictionary:
```
{{
    "Projection": [
        {{
            "Field Technical Name": "",
            "Derivation Type": "",
            "Derivation logic": "",
            "Source": "",
            "Excluded columns": ,
            "Natural Language Derivation": ,
            "JOIN KEYS": 
        }},
        ...
    ]
}}
```         

Input Dictionary: {excel_dict}

---

JSON format:
```JSON
{{
 "asset_name": "{asset_name}",
 "settings": {{
  "view-comment": "{view_comment}",
  "common_column_names": "{common_column_names}"
 }},
 "Create a property for each unique "Projection", present in the input dictionary, which has sub-string 'projection' in it and set this key as that "Projection".": {{
  "base_table": "Select the unique value of "Source" from the selected dictionary for which this "Projection" property is created.",
  "include_columns": [Create a list of all unique 'Field Technical Name', whose 'Derivation Type' is 'Direct', from all the inner dictionaries of the list for which this "Projection" property is created.],
  "exclude_columns": [Create a list of all unique 'Excluded columns', from all the inner dictionaries of the list for which this "Projection" property is created.],
  "filter": "Fill this value with 'Derivation Logic', from the selected dictionary for which this "Projection" property is created, whose 'Derivation Type' is 'where'.",
  "special_columns": [Make sure to create a list of strings, where each string is generated for below mentioned 'Derivation Types' which are part of the dictionaries considered for this "Projection" property, by following below instructions:
        1. If 'Derivation Type' is 'Rename' generate string as "Derivation logic AS Field Technical Name".
        2. If 'Derivation Type' is 'Constant' generate string as "Derivation logic AS Field Technical Name".
        3. If 'Derivation Type' is 'Derive' and 'Derivation logic' doesn't start with "if", generate string as "Derivation logic AS Field Technical Name".
        4. If 'Derivation Type' is 'Derive' and 'Derivation logic' starts with "if", first replace "if" in 'Derivation Logic' with "CASE WHEN", add "END" at the end of 'Derivation Logic' and then generate string as "Derivation logic AS Field Technical Name". Eg: 'Derivation Type'="if KOART='D' then VALUE_EUR*10/100 else 0", 'Field Technical Name'="DISCOUNT" -> "CASE WHEN KOART='D' then VALUE_EUR*10/100 else 0 END AS DISCOUNT"
        In any of the above cases consider entire values of 'Derivation logic' and 'Field Technical Name'. Do not consider dictionaries for which 'Derivation Type' is 'Direct'.]
        "group_by": "Fill this value with 'Derivation Logic', from the selected dictionary for which this "Projection" property is created, whose 'Derivation Type' is 'group by'."
 }},
    There might be multiple projections so, make sure to create the property for all projections without excluding anything which is present in the input dictionary.

 "Create a property for each unique "Projection", present in the input dictionary, which has sub-string 'custom' in it and set this key as that "Projection".": {{
  "sql_query": "Select a unique value of "Derivation logic", whose 'Derivation Type' is 'SQL', for which this "Custom" property is created.",
  "comment": "Create a string, from the unique value of "Source" for which this "Custom" property is created, by removing the sub-string "projection_" from that value.",
  "base_table": "Select the unique value of "Source" from the selected dictionary for which this "Custom" property is created."
 }},
    There might be multiple customs so, make sure to create the property for all customs without excluding anything which is present in the input dictionary.

 "Create a property for each unique "Projection", present in the input dictionary, which has sub-string 'union' in it and set this key as that "Projection".": {{
        "comment": "",
        "union_tables": [
            In the "Source" value, for which this "Union" property is created, a list of views is provided with a delimeter, generate a property as below for each view.
            {{
                "base_table": "Fill this value with unique view name from the list.",
                "include_columns": [],
                "exclude_columns": [],
                "special_columns": [],
                "filter": [],
                "column_rename": {{}},
                "union_type": "inner"
            }},
        ]
    }},
    There might be multiple unions so, make sure to create the property for all unions without excluding anything which is present in the input dictionary.
    ...
}}
```

---

Instructions:
1. There might be multiple projections, customs and unions so, make sure to create the property for all projections, customs and unions without excluding anything which is present in the input dictionary.
2. Do not hallucinate, generate the exact structure as per the given instructions. Do not disregard the instructions, generate accurate values from the given input.
3. Only provide the actual JSON structure in the response and no comments, notes or backticks (```), so that the response can be written in a JSON file.
4. Make sure complete JSON structure is generated and it doesn't get cut-off towards the end, so that the JSON parser doesn't throw errors when parsing the response.

         """
        }
    ],temperature=0) # ,max_tokens=1024
    return response.choices[0].message.content


# Main Streamlit App
def main():
    # st.set_page_config(layout="wide")
    
    # Load background image
    image = Image.open("Capgemini-1024x576 - Copy.jpg")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('data:image/jpeg;base64,{base64_image}');
                background-size: cover;
            }}      
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='white-space: nowrap;'>Lakehouse Asset Business Transformation</h1>", unsafe_allow_html=True)

    # UI Elements
    excel_file = st.file_uploader("Upload the mapping file", type=["xlsx", "xls"])
    col1, col2 = st.columns(2)
    asset_name = col1.text_input("Enter Asset Name")
    view_comment = col2.text_input("Enter Comment")

    # Session state initialization
    if "processed" not in st.session_state:
        st.session_state["processed"] = False
        st.session_state["json_final"] = None
        st.session_state["json_edit_mode"] = False
        st.session_state["json_confirmed"] = False

    # Process Mapping Sheet
    if excel_file and asset_name and view_comment:
        st.write("✅ File uploaded successfully:", excel_file.name)

        if st.button("Process mapping sheet"):
            st.session_state["processed"] = True
            st.rerun()

    # Generate JSON (Only Once)
    if st.session_state["processed"] and not st.session_state["json_final"]:
        st.write("⏳ Generating JSON... Please wait...")
        excel_dict = create_nested_dict_from_excel(excel_file)
        output_dict = process_dictionary(excel_dict)
        common_column_names = "1"
        json_response = generate_json(output_dict, asset_name, view_comment, common_column_names)
        if "```JSON" in json_response:
            json_match = re.search(r"```JSON\n(.*?)\n```", json_response, re.DOTALL)

            if json_match:
                json_response = json_match.group(1)
            else:
                print("Backticks not found!")
        # Store JSON in session state
        st.session_state["json_final"] = json_response
        st.session_state["json_edit_mode"] = False
        st.session_state["json_confirmed"] = False
        st.rerun()

    # Show Generated JSON
    if st.session_state["json_final"]:
        st.subheader("Generated JSON:")
        st.json(st.session_state["json_final"])

        # Buttons for Editing and Proceeding
        col3, col4 = st.columns(2)
        if col3.button("Update JSON"):
            st.session_state["json_edit_mode"] = True
            st.rerun()

        if col4.button("Not Required", key="json_proceed"):
            st.session_state["json_edit_mode"] = False
            st.session_state["json_confirmed"] = True
            st.rerun()

        # JSON Editing Mode
        if st.session_state["json_edit_mode"]:
            edited_json = st.text_area(
                "Edit your JSON:", st.session_state["json_final"], height=300
            )
            if st.button("Confirm JSON Update"):
                try:
                    st.session_state["json_final"] = edited_json
                    st.session_state["json_edit_mode"] = False
                    st.session_state["json_confirmed"] = True
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format! Please correct and try again.")

        # Download Button (Only When JSON is Confirmed)
        if st.session_state["json_confirmed"]:
            # json_str = json.dumps(st.session_state["json_final"], indent=4)
            #  # Create a BytesIO object to store the JSON content
            # json_file = io.BytesIO(json_str.encode("utf-8"))
            # st.download_button("📥 Download JSON", json_file, file_name="output.json", mime="application/json")
             # Download buttons

            def create_json_download(json_data):
                json_content = json.dumps(json_data, indent=4)
                output = io.BytesIO()
                output.write(json_content.encode())
                output.seek(0)
                return output
            st.download_button(
                    label="📥 Download as JSON",
                    data=create_json_download(json.loads(st.session_state["json_final"])),
                    file_name="output.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
