import pandas as pd
import re
from openai import AzureOpenAI
import streamlit as st
import io
import json
from io import BytesIO
import base64
from PIL import Image

def create_nested_dict_from_excel(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Read the relevant sheet and skip the first few rows (adjust as necessary)
    df = pd.read_excel(xls, sheet_name='Finance CORE BT View', skiprows=4)
    
    # Drop columns that are completely empty (all NaN values)
    # df = df.dropna(axis=1, how='all')
    
    # Initialize the nested dictionary
    nested_dict = {}
    
    source_table = df['Source'].iloc[0]
    
    # Iterate over the rows in the dataframe
    for _, row in df.iterrows():
        # Clean the multi-line string by removing line breaks and tabs
        # print(f"\nDerivation Logic {i}: ", repr(row['Derivation logic']))
        # cleaned_value = " ".join(row['Derivation logic'].replace("\t", "").splitlines()).strip()
        cleaned_value = re.sub(r'[\t\n\r]+', ' ', row['Derivation logic']).strip()
        cleaned_value = re.sub(r'\s+', ' ', cleaned_value)
        # print(f"\nCleaned Value {i}: ", repr(cleaned_value))

        inner_dict = {
            'Description': row['Description'],
            'Field Technical Name': row['Field Technical Name'],
            'Derivation Type': row['Derivation Type'],
            'Derivation logic': cleaned_value,
            'Filter': row['Filter'],
            'Excluded columns': row['Excluded columns'],
            'Natural Language Derivation': row['Natural Language Derivation']
        }

        if source_table in nested_dict:
            nested_dict[source_table].append(inner_dict)
        else:
            nested_dict[source_table] = [inner_dict]

    return nested_dict


def generate_query(stm_dict):
    client = AzureOpenAI(
    api_key = "836fea48dd7a402dbd8cf73f4dddb2e5",  
    api_version = "2023-05-15",
    azure_endpoint = "https://bayerbichatgpt4.openai.azure.com/"
    )

    response = client.chat.completions.create(
    model="BayerBIGPT35Turbo",    
    messages=[        
        {"role": "user", "content": f"""
You are tasked with creating a SQL `SELECT` statement, which is to be executed in postgres database, for the query generation the input is nested dictionary provided. The dictionary represents the fields and their respective derivation logic for the specified source table.

Input Dictionary: {stm_dict}

- **Source Table**: `source_table_name`
- **Fields**: Each field has a technical name, a derivation type, and a derivation logic.
Note: Derivation types are DIRECT, CONCATENATE, CONSTANT and SQL. If derivation type is SQL it means that derivation logic is already specified in SQL format for that particular field.

Using this information, please generate a `SELECT` statement that includes all the fields specified in the dictionary. Make sure to implement the appropriate derivation logic for each field.

Sample structure of provided dictionary:
```
{{
    "Source": [
        {{
            "Description": "",
            "Field Technical Name": "",
            "Derivation Type": "",
            "Derivation logic": "",
            "Filter": "",
            "Excluded columns": "",
            "Natural Language Derivation": ""
        }},
        ...
    ]
}}
```

### Generated `SELECT` Statement:
1. For each field in the dictionary, you will need to apply the derivation logic if it exists (such as CONCATENATION, FUNCTION, etc.).
2. Construct the `SELECT` statement with each derived field.
3. If no derivation logic is specified, use the field directly.
4. Only 'Field Technical Name' should be used as the alias name in the query, do not use 'Derivation Type' as the alias name.
5. WITH clause should be before the SELECT statement.
---

**Example Output (for the given dictionary structure):**
```sql
SELECT 
  CONCAT(GJAHR, MONAT) AS GJAHR_MONAT,  -- Applying CONCATENATION logic
  -- Add additional fields and derivation logic here
FROM 
  source_table_name;
```

---

Instructions:
1. Make sure complete query is generated and it doesn't get cut-off towards the end.
2. Only provide the actual sql query in the response and no comments. Also remove the strings "```" and "```sql" from the response.
3. Before providing the response, check the sql syntax and provide the final response.
"""
}
    ],temperature=0,max_tokens=1024)
    return response.choices[0].message.content

def main():
    image = Image.open("Capgemini-1024x576 - Copy.jpg")

    # Convert the image to base64
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
    # st.title("Mapping Sheet to JSON")
    st.markdown("<h1 style='white-space: nowrap;'>Lakehouse Asset Business Transformation</h1>", unsafe_allow_html=True)

    excel_file = st.file_uploader("Upload the mapping file", type=["xlsx", "xls"])

    if "processed" not in st.session_state:
        st.session_state["processed"] = False
        st.session_state["file_name"] = None
        st.session_state["query_generated"] = False
        st.session_state["query_final"] = None
        st.session_state["json_ready"] = False
        st.session_state["json_final"] = None
        st.session_state["json_edit_mode"] = False
        st.session_state["json_confirmed"] = False

    if excel_file:
        st.write("✅ File uploaded successfully:", excel_file.name)

        if st.button("Process mapping sheet"):
            st.session_state["processed"] = True
            st.session_state["file_name"] = excel_file.name

    if st.session_state["processed"]:
        #st.write("Creating source dictionary...")
        excel_dict = create_nested_dict_from_excel(excel_file)
        #st.write("Source dictionary created!")

        source_table = list(excel_dict.keys())[0]

        st.write("Generating SQL query...")
        if not st.session_state["query_generated"]:
            query = generate_query(excel_dict)
            st.session_state["query_generated"] = True
            st.session_state["query_final"] = query
        else:
            query = st.session_state["query_final"]

        st.write("Generated SQL query:")
        st.code(query, language="sql")

        # User choice to update or proceed
        col1, col2 = st.columns(2)
        update_query = col1.button("Update Query")
        proceed_query = col2.button("Not Required")

        if update_query:
            st.session_state["update_mode"] = True

        if proceed_query:
            st.session_state["update_mode"] = False
            st.session_state["json_ready"] = True

        # If update is selected, allow query editing
        if "update_mode" in st.session_state and st.session_state["update_mode"]:
            edited_query = st.text_area("Edit your SQL query:", query, height=200)
            if st.button("Confirm Update"):
                st.session_state["query_final"] = edited_query
                st.session_state["json_ready"] = True
                st.session_state["update_mode"] = False

        # Proceed to JSON generation only if the user chose "Not Required" or confirmed query update
        if st.session_state["json_ready"]:
            st.write("Generating JSON structure...")
            json_data = {
                "asset_name": "",
                "settings": {
                    "view-comment": "",
                    "common_column_names": ""
                },
                "projection-fis-beleg-view": {
                    "sql_query": st.session_state["query_final"],
                    "comment": "",
                    "base_table": source_table
                }
            }

            if not st.session_state["json_confirmed"]:
                st.session_state["json_final"] = json_data

            st.write("Generated JSON structure:")
            st.json(st.session_state["json_final"])

            # User choice to update JSON or proceed to download
            col3, col4 = st.columns(2)
            update_json = col3.button("Update JSON")
            proceed_json = col4.button("Not Required", key="json_proceed")

            if update_json:
                st.session_state["json_edit_mode"] = True

            if proceed_json:
                st.session_state["json_edit_mode"] = False
                st.session_state["json_confirmed"] = True

            # If update is selected, allow JSON editing
            if st.session_state["json_edit_mode"]:
                edited_json = st.text_area(
                    "Edit your JSON:", json.dumps(st.session_state["json_final"], indent=4), height=300
                )
                if st.button("Confirm JSON Update"):
                    try:
                        st.session_state["json_final"] = json.loads(edited_json)
                        st.session_state["json_confirmed"] = True
                        st.session_state["json_edit_mode"] = False
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format! Please correct the JSON and try again.")

            # Proceed to download only if the user chose "Not Required" or confirmed JSON update
            if st.session_state["json_confirmed"]:
                # Download buttons
                def create_download(content, filename, mime):
                    output = io.BytesIO()
                    output.write(content.encode())
                    output.seek(0)
                    return output

                st.download_button(
                    label="Download as SQL",
                    data=create_download(st.session_state["query_final"], "output.sql", "application/sql"),
                    file_name="output.sql",
                    mime="application/sql"
                )

                st.download_button(
                    label="Download as JSON",
                    data=create_download(json.dumps(st.session_state["json_final"], indent=4), "output.json", "application/json"),
                    file_name="output.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
