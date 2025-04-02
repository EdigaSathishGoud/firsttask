import pandas as pd
import re
import streamlit as st
import io
import json
from io import BytesIO
import base64
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
        filter = str(row['Filter'])
        
        # Clean all multi-line strings by removing line breaks and tabs
        # print(f"\nDerivation Logic {i}: ", repr(row
        cleaned_derivation_logic = re.sub(r'\s+', ' ', derivation_logic.strip())
        cleaned_filter = re.sub(r'\s+', ' ', filter.strip())
        # print(f"\nCleaned Value {i}: ", repr(cleaned_value))

        Projection = row['Projection']

        inner_dict = {
            'Description': row['Description'],
            'Field Technical Name': row['Field Technical Name'],
            'Derivation Type': row['Derivation Type'],
            'Derivation logic': cleaned_derivation_logic,
            'Source': row['Source'],
            'Filter': cleaned_filter,
            'Excluded columns': row['Excluded columns'],
            'Natural Language Derivation': row['Natural Language Derivation'],
            'JOIN KEYS': row['JOIN KEYS']
        }

        if Projection in nested_dict:
            nested_dict[Projection].append(inner_dict)
        else:
            nested_dict[Projection] = [inner_dict]

    return nested_dict

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
            "Description": NaN,
            "Field Technical Name": "",
            "Derivation Type": "",
            "Derivation logic": "",
            "Source": "",
            "Filter": "",
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
		"include_columns": [Create a list of all unique 'Field Technical Name', whose 'Derivation Type' is 'Direct', using all the inner dictionaries of the list for which this "Projection" property is created.],
		"exclude_columns": [Create a list of all unique 'Excluded columns', using all the inner dictionaries of the list for which this "Projection" property is created.],
		"filter": "Create a single string by combining all unique 'Filter', using all the inner dictionaries of the list for which this "Projection" property is created, using 'AND' operator.",
		"special_columns": [Make sure to create a list of strings, where each string is generated for below mentioned 'Derivation Types' which are part of the dictionaries considered for this "Projection" property, by following below instructions:
        1. If 'Derivation Type' is 'rename' generate string as "Derivation logic as Field Technical Name".
        2. If 'Derivation Type' is 'Constant' generate string as "'Derivation logic' as Field Technical Name".
        3. If 'Derivation Type' is 'Derive', first replace "if" in 'Derivation Logic' with "case when", add "end" at the end of 'Derivation Logic' and then generate string as "Derivation logic as Field Technical Name".
        In any of the above cases consider entire values of 'Derivation logic' and 'Field Technical Name'.
        Eg: 'Derivation Type'="if KOART='D' then VALUE_EUR*10/100 else 0", 'Field Technical Name'="DISCOUNT" -> "case when KOART='D' then VALUE_EUR*10/100 else 0 end as DISCOUNT"]
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
2. While generating lists in the JSON structure, make sure not to put each element of the list on a new line. Create a single-line list rather than a multi-line list.
3. Do not hallucinate, generate the exact structure as per the given instructions. Do not disregard the instructions, generate accurate values from the given input.
4. Only provide the actual JSON structure in the response and no comments, notes or backticks (```), so that the response can be written in a JSON file.
5. Make sure complete JSON structure is generated and it doesn't get cut-off towards the end, so that the JSON parser doesn't throw errors when parsing the response.

         """
        }
    ],temperature=0) # ,max_tokens=1024
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
    
    st.markdown("<h1 style='white-space: nowrap;'>Lakehouse Asset Business Transformation</h1>", unsafe_allow_html=True)
   
    excel_file = st.file_uploader("Upload the mapping file", type=["xlsx", "xls"])
    col3, col4 = st.columns(2)
    asset_name = col3.text_input("Enter Asset_name")
    view_comment = col4.text_input ("Enter Comment")
    # asset_name = st.text_input("Enter Asset_name")
    # view_comment = st.text_input ("Enter Comment")
    if "processed" not in st.session_state:
        st.session_state["processed"] = False
        st.session_state["json_final"] = None
        st.session_state["json_edit_mode"] = False
        st.session_state["json_confirmed"] = False
    
    if excel_file and asset_name and view_comment:
        st.write("✅ File uploaded successfully:", excel_file.name)
        
        if st.button("Process mapping sheet"):
            st.session_state["processed"] = True
    
    if st.session_state["processed"]:
        st.write("Json generation started ...")
        excel_dict = create_nested_dict_from_excel(excel_file)
        # source_table = list(excel_dict.keys())[0]
        common_column_names ="1"
        response = generate_json(excel_dict, asset_name, view_comment, common_column_names)
        if "```JSON" in response:
            json_match = re.search(r"```JSON\n(.*?)\n```", response, re.DOTALL)

            if json_match:
                response = json_match.group(1)
            else:
                print("Backticks not found!")
    
        
        if not st.session_state["json_confirmed"]:
            st.session_state["json_final"] = response
        
        st.write("Generated JSON structure:")
        st.json(st.session_state["json_final"])
        
        col1, col2 = st.columns(2)
        update_json = col1.button("Update JSON")
        proceed_json = col2.button("Not Required", key="json_proceed")
        
        if update_json:
            st.session_state["json_edit_mode"] = True
        
        if proceed_json:
            st.session_state["json_edit_mode"] = False
            st.session_state["json_confirmed"] = True
        
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
        
        if st.session_state["json_confirmed"]:
            def create_download(content, filename, mime):
                output = io.BytesIO()
                output.write(content.encode())
                output.seek(0)
                return output
            
            st.download_button(
                label="Download as JSON",
                data=create_download(json.dumps(st.session_state["json_final"], indent=4), "output.json", "application/json"),
                file_name="output.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
