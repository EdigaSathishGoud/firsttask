import pandas as pd
import re
from openai import AzureOpenAI
import os
import json
import streamlit as st
import io
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
    # image = Image.open("Capgemini-1024x576 - Copy.jpg")

    # # Convert the image to base64
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG")
    # base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # st.markdown(
    #     f"""
    #     <style>
    #         .stApp {{
    #             background-image: url('data:image/jpeg;base64,{base64_image}');
    #             background-size: cover;
    #         }}      
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.title("Mapping Sheet to JSON")

    excel_file = st.file_uploader("Upload the mapping file", type=["xlsx","xls"])
    # excel_file = "Finance Use Case_Requirement Specs v1.0.xlsx"

    # folder_path = os.path.dirname(os.path.abspath(__file__))

    # Initialize session state variables if not already set
    if "processed" not in st.session_state:
        st.session_state["processed"] = False
        st.session_state["file_name"] = None

    if excel_file:
        st.write("✅ File uploaded successfully:", excel_file.name)
        
        if st.button("Process mapping sheet"):
            st.session_state["processed"] = True
            st.session_state["file_name"] = excel_file.name

    if st.session_state["processed"]:
        st.write("Creating source dictionary...")
        excel_dict = create_nested_dict_from_excel(excel_file)
        # print("\nSource Dictionary:\n")
        # print(json.dumps(excel_dict, indent=4))
        st.write("Source dictionary created!")

        for key in excel_dict:
            # print("\n",key)
            source_table = key
            break

        st.write("Generating SQL query...")
        query = generate_query(excel_dict)
        # print(f"\n\nSelect Query:\n{query}\n\n")
        st.write("Generated SQL query: ")
        st.code(query, language="sql")

        # Create the dictionary with the desired structure
        st.write("Generating JSON structure...")
        json_data = {
            "asset_name": "",
            "settings": {
                "view-comment": "",
                "common_column_names": ""
            },
            "projection-fis-beleg-view": {
                "sql_query": query,
                "comment": "",
                "base_table": source_table
            }
        }

        # print(json.dumps(json_data, indent=4))
        st.write("Generated JSON structure: ")
        st.json(json_data)

        # Write the data to a JSON file
        # with open('new_output.json', 'w') as json_file:
        #     json.dump(json_data, json_file, indent=4)
        
        # st.write(f"\n\nData stored in JSON file: {folder_path}\\new_output.json")
        # print(f"\n\nData stored in JSON file: {folder_path}\\new_output.json")

        # Create downloadable content in .sql format
        def create_sql_download(sql_content):
            output = io.BytesIO()
            output.write(sql_content.encode())
            output.seek(0)
            return output

        # Create downloadable content in .json format
        def create_json_download(json_data):
            json_content = json.dumps(json_data, indent=4)
            output = io.BytesIO()
            output.write(json_content.encode())
            output.seek(0)
            return output

        # Download buttons
        st.download_button(
            label = "Download as SQL",
            data = create_sql_download(query),
            file_name = "output.sql",
            mime = "application/sql",
            key = "sql_download"
        )

        st.download_button(
            label = "Download as JSON",
            data = create_json_download(json_data),
            file_name = "output.json",
            mime = "application/json",
            key = "json_download"
        )


if __name__ == "__main__":
    main()