import pandas as pd
import re
import streamlit as st
import io
import json
import base64
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_model = os.getenv("OPENAI_MODEL")


def initialize_openai_client():
    client = AzureOpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),  
    api_version = os.getenv("OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")
    )
    return client


def create_nested_dict_from_excel(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Read the relevant sheet and skip the first few rows (adjust as necessary)
    df = pd.read_excel(xls) # , sheet_name='Sheet1', skiprows=4

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
            'right project name /table name': "",
            'Join type ': "",
            'column_prefix': "",
            'Join Include Columns': "",
            'Excluded columns': row['Excluded columns'],
            'Join filter': "",
            'Special Columns': "",
            # 'Natural Language Derivation': row['Natural Language Derivation'],
            # 'JOIN KEYS': row['JOIN KEYS']
        }

        if Projection in nested_dict:
            nested_dict[Projection].append(inner_dict)
        else:
            nested_dict[Projection] = [inner_dict]

    return nested_dict


def llm_call_for_projection(derivation_logic):
    """
    Function to call LLM and generate a query based on the derivation logic.
    Replace this with actual API calls to your LLM (e.g., Azure OpenAI).
    """
    client = initialize_openai_client()
    response = client.chat.completions.create(
    model = openai_model, 
        messages=[
            {"role": "system", "content": """You are an expert in writing conditions or operations in Databricks SQL syntax for the logic provided in natural language by following below instructions:
                1. Understand the entire logic provided in natural language and generate a technical condition using appropriate Databricks syntax.
                2. Convert natural language into actual technical conditions by following Databricks executable SQL syntax. Make sure to use Databricks supported operators (=, <>, >, <) and functions for the conversion. Do not use keywords like 'INTERVAL', 'SEQUENCE' which are not supported in Databricks.
                3. Only provide the actual databricks SQL condition in the response and no comments/notes. Also remove unnecessary backticks like "```" and "```sql" from the response.

                Examples:
                1. If natural language is "Concatenation of a, b and c" then output should only contain: "CONCAT(a,b,c)"
                2. If natural language is "Select records where the PERIOD_BUCH is derived from the previous month's date, formatted as  yyyyMM  if the month is October or later, or yyyy0M if it is January to September. Additionally, ensure that AM_FLAG is not blank and OPTYPE is not 'D'" then output should only contain:\n"PERIOD_BUCH = DATE_FORMAT(ADD_MONTHS(current_date(), -1), 'yyyyMM') AND AM_FLAG <> '' AND OPTYPE <> 'D'"
             """},
            {"role": "user", "content": f"Generate a Databricks SQL query condition for the following logic provided in natural language:\n{derivation_logic}"}
            
        ],temperature=0)
    # 3. Do not include Select , from , table names etc in the output . Provide only the condition which satisfies the given input.
    return response.choices[0].message.content


def llm_call_for_custom(derivation_logic):
    client = initialize_openai_client()
    response = client.chat.completions.create(
    model = openai_model, 
        messages=[
            {"role": "system", "content": "You're an expert in writing complete Databricks SQL queries from logic provided in natural language. Ensure that you follow the below instructions while generating the response:\n1. First understand the entire logic provided in natural language and then generate the complete query using appropriate Databricks syntax.\n2. Only use Databricks supported operators and functions, do not use keywords which are not supported in Databricks.\n3. Use the exact table name provided in natural language as it is in the response(For example, entire sub-string like 'projection-base_bill_of_payments' before the word 'table' is actual source table name).\n4. If you're adding necessary column names in the 'GROUP BY' clause, do not add the keyword 'ALL'.\n5. Entire query should be generated, it shouldn't get cut-off towards the end.\n6. Please do not add any comments, explanations, notes or backticks to the response.\n7. Do not generate python code, strictly generate Databricks SQL query."},
            {"role": "user", "content": f"Generate Databricks SQL query for the following logic provided in natural language:\n{derivation_logic}"}
        ],temperature=0)
    return response.choices[0].message.content


def llm_call_for_join(derivation_logic):
    client = initialize_openai_client()
    response = client.chat.completions.create(
    model = openai_model, 
        messages=[
            {"role": "system", "content": """
            You are given a natural language description of a Databricks SQL join operation with filter conditions. Your task is to convert the description into a dictionary format with the following structure:
                "Source": The name of the left table or project.
                "right project name /table name": The name of the right table or project.
                "Join type": The type of join used, e.g., "inner".
                "Join filter": Convert filter conditions provided in natural language into actual technical conditions by following Databricks executable SQL syntax. Make sure to use Databricks supported operators (=, <>, >, <) and functions to convert the filters. Do not use keywords like 'INTERVAL', 'SEQUENCE' which are not supported in Databricks. Then combine them into a single string, using "AND" operator, to generate the join filter.
                "JOIN KEYS": String of columns used for the join in key-value pairs. Do not add curly brackets in this value.
                "column_prefix": Prefix to be added to columns of right table,
                "Join Include Columns": Columns to be included from the left table,
                "Excluded columns": Columns to be excluded from the left table,
                "Special Columns": Special columns to be included from the left table,

            For example, the natural language description is converted into the following format:
            {
                "Source": "A",
                "right project name /table name": "B",
                "Join type ": "inner",
                "Join filter": "str",
                "JOIN KEYS": ""a1": "b1", "a2": "b2", "a3": "b3"",
                "column_prefix": "str",
                "Join Include Columns": "str",
                "Excluded columns": "str",
                "Special Columns": "str",
            }
            """},
            {"role": "user", "content": f"Generate a dictionary in the provided format for the following natural language description:\n{derivation_logic}"}
            
        ],temperature=0)
    return response.choices[0].message.content


def process_dictionary(data):
    # print("entered process_dictinary function")
    for key, values in data.items():
        # print("Key is :\n",key,"\nvalues are :\n",values)
        for entry in values:
            derivation_type = str(entry.get("Derivation Type")).lower()
            if derivation_type == "filter":
                derivation_logic = entry.get("Derivation logic")
                # print(f"Derivation_logic for table ,{key} is:",derivation_logic)
                if derivation_logic and derivation_logic.lower() != "nan":
                    # print("calling llm:\n")
                    query = llm_call_for_projection(derivation_logic)
                    query = query.replace("\n"," ")

                    # Replace multiple spaces with a single space
                    query = re.sub(r'\s+', ' ', query)

                    # print("\nllm response: \n",query)
                    entry["Derivation logic"] = query
            
            if derivation_type == "derive":
                derivation_logic = entry.get("Derivation logic")
                if derivation_logic and derivation_logic.lower() != "nan":
                    query = llm_call_for_projection(derivation_logic)
                    query = query.replace("\n"," ")

                    # Replace multiple spaces with a single space
                    query = re.sub(r'\s+', ' ', query)

                    # print("\nllm response: \n",query)
                    entry["Derivation logic"] = query
            
            if derivation_type == "sql":
                derivation_logic = entry.get("Derivation logic")
                if derivation_logic and derivation_logic.lower() != "nan":
                    print("\nSQL in natural language: \n",derivation_logic)
                    query = llm_call_for_custom(derivation_logic)
                    query = query.replace("\n"," ")

                    # Replace multiple spaces with a single space
                    query = re.sub(r'\s+', ' ', query)

                    print("\nSQL in Databricks syntax: \n",query)
                    entry["Derivation logic"] = query
            
            if derivation_type == "join":
                derivation_logic = entry.get("Derivation logic")
                if derivation_logic and derivation_logic.lower() != "nan":
                    str_dict = llm_call_for_join(derivation_logic)

                    # Convert the response string into a dictionary
                    response_dict = json.loads(str_dict)
                    
                    # Define the target dictionary with required keys
                    entry["Derivation logic"] = ""
                    entry["Source"] = response_dict.get("Source", "")
                    entry["right project name /table name"] = response_dict.get("right project name /table name", "")
                    entry["Join type "] = response_dict.get("Join type", "")
                    entry["column_prefix"] = response_dict.get("column_prefix", "")
                    entry["Join Include Columns"] = response_dict.get("Join Include Columns", "")
                    entry["Excluded columns"] = response_dict.get("Excluded columns", "")
                    entry["Join filter"] = response_dict.get("Join filter", "")
                    entry["Special Columns"] = response_dict.get("Special Columns", "")
                    entry["JOIN KEYS"] = response_dict.get("JOIN KEYS", "")

    return data


def generate_json(excel_dict):
    client = initialize_openai_client()
    response = client.chat.completions.create(
    model = openai_model,
    messages=[
        {"role": "user", "content": f"""
You are a Databricks SQL and JSON expert. I want you to generate JSON structure in the exact format as provided below. Inside the format, I've provided instructions on how to generate the values for respective keys. Refer the data provided in the dictionary according to the given instructions.

---

Sample structure of provided dictionary:
```
{{
    "Projection": [
        {{
            'Field Technical Name': "",
            'Derivation Type': "",
            'Derivation logic': "",
            'Source': "",
            'right project name /table name': "",
            'Join type ': "",
            'column_prefix': "",
            'Join Include Columns': "",
            'Excluded columns': "",
            'Join filter': "",
            'Special Columns': "",
            'JOIN KEYS': ""
        }},
        ...
    ]
}}
```         

Input Dictionary: {excel_dict}

---

JSON format:
```JSON
"Create a property for each unique "Projection", present in the input dictionary, which has sub-string 'projection' in it and set this key as that "Projection".": {{
    "base_table": "Select the unique value of "Source" from the selected dictionary for which this "Projection" property is created.",
    "include_columns": [Create a list of all unique 'Field Technical Name', whose 'Derivation Type' is 'Direct', from all the inner dictionaries of the list for which this "Projection" property is created.],
    "exclude_columns": [Create a list of all unique 'Excluded columns', from all the inner dictionaries of the list for which this "Projection" property is created.],
    "filter": "Fill this value with 'Derivation Logic', from the selected dictionary for which this "Projection" property is created, whose 'Derivation Type' is 'filter'.",
    "special_columns": [Make sure to create a list of strings, where each string is generated for below mentioned 'Derivation Types' which are part of the dictionaries considered for this "Projection" property, by following below instructions:
    1. If 'Derivation Type' is 'Rename' generate string as "Derivation logic AS Field Technical Name".
    2. If 'Derivation Type' is 'Constant' and 'Derivation logic' has opening and closing single quotes generate string as "Derivation logic AS Field Technical Name".
    3. If 'Derivation Type' is 'Constant' and 'Derivation logic' doesn't have opening and closing single quotes generate string as "'Derivation logic' AS Field Technical Name".
    4. If 'Derivation Type' is 'Derive' and 'Derivation logic' doesn't start with "if", generate string as "Derivation logic AS Field Technical Name".
    5. If 'Derivation Type' is 'Derive' and 'Derivation logic' starts with "if", first replace "if" in 'Derivation Logic' with "CASE WHEN", add "END" at the end of 'Derivation Logic' and then generate string as "Derivation logic AS Field Technical Name". Eg: 'Derivation Type'="if KOART='D' then VALUE_EUR*10/100 else 0", 'Field Technical Name'="DISCOUNT" -> "CASE WHEN KOART='D' then VALUE_EUR*10/100 else 0 END AS DISCOUNT"
    In any of the above cases consider entire values of 'Derivation logic' and 'Field Technical Name'. Do not consider dictionaries for which 'Derivation Type' is 'Direct'.]
    "group_by": "Fill this value with 'Derivation Logic', from the selected dictionary for which this "Projection" property is created, whose 'Derivation Type' is 'group by'."
}}
There might be multiple projections so ensure that properties are created for all projections without excluding any projection / elements present in the input dictionary.

"Create a property for each unique "Projection", present in the input dictionary, which has sub-string 'custom' in it and set this key as that "Projection".": {{
    "sql_query": "Select a unique value of "Derivation logic", whose 'Derivation Type' is 'SQL', for which this "Custom" property is created.",
    "comment": "Create a string, from the unique value of "Source" for which this "Custom" property is created, by removing the sub-string "projection_" from that value.",
    "base_table": "Select the unique value of "Source" from the selected dictionary for which this "Custom" property is created."
}}
There might be multiple customs so ensure that properties are created for all customs without excluding any projection / elements present in the input dictionary.

"Create a property for each unique "Projection", present in the input dictionary, which has sub-string 'union' in it and set this key as that "Projection".": {{
    "comment": "",
    "union_tables": [
        Generate a property as below for each unique "Source" from all the inner dictionaries of the list for which this "Join" property is created.
        {{
            "base_table": "Fill this value with the unique "Source" value",
            "include_columns": [],
            "exclude_columns": [],
            "special_columns": [],
            "filter": [],
            "column_rename": {{}},
            "union_type": "Fill this value with 'Derivation Logic', from the selected inner dictionary for which this property is created, whose 'Derivation Type' is 'union all'"
        }},
    ]
}}
There might be multiple unions so ensure that properties are created for all unions without excluding any projection / elements present in the input dictionary.

"Create a property for each unique "Projection", present in the input dictionary, which has sub-string 'join' in it and set this key as that "Projection".": {{
    "base_table": "Select the unique value of "Source" from the selected dictionary for which this "Join" property is created.",
    "comment": "",
    "join_tables": [
        Generate a property as below for each unique 'right project name /table name' from all the inner dictionaries of the list for which this "Join" property is created.
        {{
            "base_table": "Fill this value with unique 'right project name /table name' value",
            "include_columns": [Create a list of all unique values present in the 'Join Include Columns', which are seperated by a delimeter, from the selected inner dictionary for which this property is created.],
            "exclude_columns": [Create a list of all unique values present in the 'Excluded columns', which are seperated by a delimeter, from the selected inner dictionary for which this property is created.],
            "special_columns" :[Create a list of strings from all unique values present in the 'Special Columns', which are seperated by a delimeter, from the selected inner dictionary for which this property is created.],
            "join_keys": {{Fill this dictionary with unique key-value pairs present in the 'JOIN KEYS', which are seperated by a delimeter, from the selected inner dictionary for which this property is created.}},
            "filter": "Fill this value with unique 'Join filter', from the selected inner dictionary for which this property is created.",
            "join_type": "Fill this value with unique 'Join type ', from the selected inner dictionary for which this property is created.",
            "alias": "",
            "column_prefix":"Fill this value with unique 'column_prefix', from the selected inner dictionary for which this property is created."
        }}, 
    ]
}}
There might be multiple joins so ensure that properties are created for all joins without excluding any projection / elements present in the input dictionary.
```

---

Instructions:
1. There might be multiple projections, customs, unions and joins so ensure that properties are created for all projections, customs, unions and joins without excluding any projection / elements present in the input dictionary.
2. Make sure to name the projections in the JSON excatly as "Projection"  given in the input dictionary.
3. Make sure to name the base tables in the JSON excatly as "Source" value given in the input dictionary.
4. Do not set "nan" as a value anywhere in the JSON response.
5. Do not generate a particular value in the JSON response if corresponding value in input dictionary is nan.
6. Do not hallucinate, generate the exact structure as per the given instructions. Do not disregard the instructions, generate accurate values from the given input.
7. Make sure complete JSON structure is generated and it doesn't get cut-off towards the end, so that the JSON parser doesn't throw errors when parsing the response.
8. Ensure that the generated response is exactly in the provided JSON format. Please do not add any comments, explanations or notes.
9. Please do not generate python code, make sure to generate JSON response.

         """
        }
    ],temperature=0) # ,max_tokens=1024
    # 2. Do not create the key-value pair itself, if no value is available for a particular key.
    # 6. Generate the response exactly in the provided JSON format. Do not add outermost opening and closing curly brackets as they are not present in the format. They are added later on to the response.
    # 7. Only provide the actual JSON structure in the response and no comments, notes or backticks (```), so that the response can be written to a JSON file.
    
    
    return response.choices[0].message.content


def clean_json_string(input_string):
    # Step 1: Remove the opening ```JSON and closing ```
    if "```JSON" in input_string:
        json_match = re.search(r"```JSON\n(.*?)\n```", input_string, re.DOTALL)

        if json_match:
            cleaned_string = json_match.group(1)
    
    # Step 2: Remove only the first `{` and last `}`
    if cleaned_string.startswith('{'):
        cleaned_string = cleaned_string[1:]  # Remove the first '{'
    if cleaned_string.endswith('}'):
        cleaned_string = cleaned_string[:-1]  # Remove the last '}'
    
    # Step 3: Return cleaned string preserving indentation
    return cleaned_string


def final_json(output_dict, asset_name, view_comment):
    common_column_names = "1"

    json_response = f"""
{{
    "asset_name": "{asset_name}",
    "settings": {{
        "view-comment": "{view_comment}",
        "common_column_names": "{common_column_names}"
    }}"""
    
    for key, item in output_dict.items():
        input_dict = {}
        input_dict[key] = item

        response = generate_json(input_dict)
        print(f"\nResponse for {key}:\n{response}\n")

        response = clean_json_string(response)
        
        # Check if the string ends with a newline and remove only the last one
        if response.endswith('\n'):
            response = response[:-1]
        
        # Append with a comma and new line
        json_response += "," + response
    
    json_response += "\n}"
    
    print("\nFinal response generated")
    with open("json_response.txt", 'w') as file5:
        # json.dump(data, json_file, indent=4)
        file5.write(json_response)
    
    return json_response


# def generate_sql():
#     return "SELECT * FROM dummy_table;"

def source_clean(name):
    if "." in name:
        formatted_name = name
    else:
        formatted_name = f"`{name}`"
    return formatted_name

def proj_sql(processed_dict):       
    tech_name = []
    source_table_name = []
    filter = []    
    for i in processed_dict:        
        if str(i['Derivation Type']).lower() == 'direct':
            tech_name.append(i['Field Technical Name'] )
        elif str(i['Derivation Type']).lower() == "constant" or str(i['Derivation Type']).lower() == "rename" or str(i['Derivation Type']).lower() == "derive":
            if str(i['Derivation Type']).lower() == "derive":
                cleaned_derivation_logic = i['Derivation logic'].strip('"').strip("'")
            else:
                cleaned_derivation_logic = i['Derivation logic']
            tech_name.append(f"{cleaned_derivation_logic} AS {i['Field Technical Name']}")
        source_table_name.append(i['Source'])
        if i["Derivation Type"] == "filter":
            filter.append(i['Derivation logic'])            
    tech_name = pd.Series(tech_name).dropna().tolist()    
    source_name = list(set(pd.Series(source_table_name).dropna().tolist() ))
    cleaned_source_name = source_clean(source_name[0])
    filter = list(set(pd.Series(filter).dropna().tolist()))  
    if len(filter) > 0:
        clean_filter = filter[0].strip('"')  
    join_col = ", ".join(x for x in tech_name)
    if len(filter) > 0 and len(join_col) > 0:
        proj_qry = f"SELECT {join_col} FROM {cleaned_source_name} WHERE {clean_filter}" 
    elif len(join_col) < 1:
        proj_qry = f"SELECT * FROM {cleaned_source_name}"
    elif len(join_col) > 0:        
        proj_qry = f"SELECT {join_col} FROM {cleaned_source_name}"        
    return proj_qry
    
def custom_sql(processed_dict):    
    for val in processed_dict:        
        custom_qry = val['Derivation logic'].strip("'").strip('"')
    return custom_qry   

def union_sql(processed_dict):    
    src_val = []    
    for val in processed_dict:
        src_val.append( val["Source"])
    query_parts = [f"SELECT * FROM {table}" if "." in table else f"SELECT * FROM `{table}`" for table in src_val]    
    final_query = " UNION ALL ".join(query_parts)    
    return final_query

def join_indiv_query(val):
    src = val['Source']
    right_tab_name = val['right project name /table name']
    formatted_right_table_name = source_clean(right_tab_name)
    join_type = val['Join type '].strip()
    join_include_columns = val['Join Include Columns']
    join_filter = val['Join filter']
    spl_col = val['Special Columns']
    prefix = val['column_prefix']  
    join_key = val['JOIN KEYS']
    join_key = json.loads(f'{{{join_key}}}')   
    join_type_map = {"inner":"inner join","left":"left join","right":"right join"} 
    join_key_in_select_col = [x for x in join_key.keys()]    
    join_key_in_sel = " ,".join(cond for cond in join_key_in_select_col)
    join = []
    for i,j in join_key.items():
        join.append(f"`{src}`.{i} = {prefix}.{j}")
    join_cond = " AND ".join(cond for cond in join) 
    if len(join_filter) > 0 and len(spl_col) > 0:
        join_qry = join_type_map[join_type] + f" ( SELECT {join_include_columns}, {spl_col}, {join_key_in_sel} FROM {formatted_right_table_name} WHERE {join_filter}) {prefix} ON {join_cond}"
    elif len(spl_col) > 1  and len(join_filter) < 1:
        join_qry = join_type_map[join_type] + f" ( SELECT {join_include_columns}, {spl_col}, {join_key_in_sel} FROM {formatted_right_table_name}) {prefix} ON {join_cond}"
    else:
        join_qry = join_type_map[join_type] + f" ( SELECT {join_include_columns}, {join_key_in_sel} FROM {formatted_right_table_name}) {prefix} ON {join_cond}"
    return join_qry, src, spl_col, join_include_columns,prefix


def spl_col_retrieve(spl_f):
    # cc = []
    # for i in spl_f:
    v = spl_f.split("AS")
    if len(v)>1:
    #     cc.append(v[1].strip())
    # dd = ", ".join(cc)
        return v[1].strip()
    
    
def join_sql(processed_dict,proj):            
    join_cond = []
    src_f = []
    spl_f = [] 
    incl_f = []   
    for val in processed_dict:               
        query,src,spl, join_include_columns,prefix = join_indiv_query(val)                
        # print("spl : ",spl)
        join_cond.append(query)        
        src_f.append(src)
        cleaned_spl = spl_col_retrieve(spl)
        if cleaned_spl is not None:
            spl_f.append(f"{prefix}.{cleaned_spl} AS {prefix}_{cleaned_spl}")
        incl_f.append(f"{prefix}.{join_include_columns} AS {prefix}_{join_include_columns}")
        # print("spl : ",spl_f,"\n",incl_f)
    incl_fin_joined = ", ".join(incl_f)
    final_join = " ".join(join_cond)
    finl_src = list(set(src_f))
    formatted_finl_src = source_clean(finl_src[0])
    # finl_spl = spl_col_retrieve(spl_f)
    finl_spl = ", ".join(spl_f)
    final_build_qry = f"SELECT {formatted_finl_src}.*, {incl_fin_joined}, {finl_spl} FROM {formatted_finl_src} {final_join}"        
    # print("final_build_qry : ",final_build_qry)
    return final_build_qry


def sql_from_mapping(processed_dict):
    processed_dict_keys = list(processed_dict.keys())
    fin_proj = {}
    for proj in processed_dict_keys:
        if proj.startswith("projection"):
            fin_sql = proj_sql(processed_dict[proj])
        elif proj.startswith("custom"):
            fin_sql = custom_sql(processed_dict[proj])
        elif proj.startswith("union"):
            fin_sql = union_sql(processed_dict[proj])
        elif proj.startswith("join"):
            fin_sql = join_sql(processed_dict[proj],proj)
        fin_proj[proj] = fin_sql    
    formed_sql = []
    for key,val in fin_proj.items():
        formed_sql.append(f"`{key}` AS (\n{val}\n)")
    # last_proj_name = list(fin_proj.keys())[-1]
    last_proj_name = source_clean(list(fin_proj.keys())[-1])
    join_formed_sql = "WITH\n"+",\n".join(formed_sql)+f"\nSELECT * FROM {last_proj_name}"
    return join_formed_sql


# Main Streamlit App
def main():
    st.set_page_config(layout="wide")
    
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
        st.session_state["output_type"] = None
        st.session_state["output_data"] = None
        st.session_state["edit_mode"] = False
        st.session_state["confirmed"] = False
        # st.session_state["json_final"] = None
        # st.session_state["json_edit_mode"] = False
        # st.session_state["json_confirmed"] = False

    # Process Mapping Sheet
    if excel_file and asset_name and view_comment:
        st.write("✅ File uploaded successfully:", excel_file.name)

        if st.button("Process mapping sheet"):
            st.session_state["processed"] = True
            st.session_state["output_type"] = None
            st.session_state["output_data"] = None
            st.rerun()

    # Generate JSON (Only Once)
    if st.session_state["processed"] and not st.session_state["output_type"]:
        #st.write("⏳ Generating JSON... Please wait...")
        st.subheader("Please select response format: JSON or SQL")
        col3, col4 = st.columns(2)
        
        if col3.button("Generate JSON"):
            st.write("⏳ Generating JSON... Please wait...")
            excel_dict = create_nested_dict_from_excel(excel_file)
            print("\nInput dictionary generated")

            output_dict = process_dictionary(excel_dict)
            print("\nUpdated dictionary generated")
            
            with open("Updated dictionary.txt", 'w') as file:
                json.dump(output_dict, file, indent=4)

            json_response = final_json(output_dict, asset_name, view_comment)

            # Store JSON in session state
            st.session_state["output_type"] = "json"
            st.session_state["output_data"] = json_response
            st.rerun()
        
        if col4.button("Generate SQL"):
            st.write("⏳ Generating SQL... Please wait...")
            excel_dict = create_nested_dict_from_excel(excel_file)
            print("\nInput dictionary generated")

            output_dict = process_dictionary(excel_dict)
            print("\nUpdated dictionary generated")
            
            with open("Updated dictionary.txt", 'w') as file:
                json.dump(output_dict, file, indent=4)

            fin_proj = sql_from_mapping(output_dict)

            st.session_state["output_type"] = "sql"
            st.session_state["output_data"] = fin_proj
            st.rerun()

    # Show Generated JSON and SQL
    if st.session_state["output_data"]:
        st.subheader(f"Generated {st.session_state['output_type'].upper()}:")
        if st.session_state["output_type"] == "json":
            st.json(st.session_state["output_data"])
        else:
            st.code(st.session_state["output_data"], language="sql")

        # Buttons for Editing and Proceeding
        col5, col6 = st.columns(2)
        if col5.button("Update Output"):
            st.session_state["edit_mode"] = True
            st.rerun()

        if col6.button("Not Required", key="proceed"):
            st.session_state["edit_mode"] = False
            st.session_state["confirmed"] = True
            st.rerun()

        # JSON Editing Mode
        if st.session_state["edit_mode"]:
            edited_output = st.text_area(
                "Edit your Output:", st.session_state["output_data"], height=300
            )
            if st.button("Confirm Update"):
                st.session_state["output_data"] = edited_output
                st.session_state["edit_mode"] = False
                st.session_state["confirmed"] = True
                st.rerun()

        # Download Button (Only When JSON is Confirmed)
        if st.session_state["confirmed"]:
            
            def create_download(output_data, output_type):
                output = io.BytesIO()
                if output_type == "json":
                    content = json.dumps(json.loads(output_data), indent=4)
                    file_name = "output.json"
                    mime = "application/json"
                else:
                    content = output_data
                    file_name = "output.sql"
                    mime = "text/sql"
                output.write(content.encode())
                output.seek(0)
                return output, file_name, mime
            
            download_file, download_name, download_mime = create_download(st.session_state["output_data"], st.session_state["output_type"])
            st.download_button(
                label=f"📥 Download {st.session_state['output_type'].upper()}",
                data=download_file,
                file_name=download_name,
                mime=download_mime
            )

if __name__ == "__main__":
    main()
