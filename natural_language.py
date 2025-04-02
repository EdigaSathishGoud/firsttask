import pandas as pd
import json
import os
from openai import AzureOpenAI
from pprint import pprint
import re


def create_nested_dict_from_excel(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Read the relevant sheet and skip the first few rows (adjust as necessary)
    df = pd.read_excel(xls, sheet_name='Sheet1') # , skiprows=4

    # Drop rows where all values are NaN
    df_cleaned = df.dropna(how='all')

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
                        print("No valid SELECT expression found.")
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
             If Derivation logic is "Concatenation of a, b and c" then output should only contain Concat(a,b,c)
             Don not include Select , from , table names etc in the output . Provide only the condition which satisfies the given input.
             Dont include ```sql at the beginning and ``` at the end of the generated output
             """},
            {"role": "user", "content": f"Generate an SQL query condition for the following logic: {derivation_logic}"}
            
        ],temperature=0)
    return response.choices[0].message.content

def main():
    excel_file = r"C:\Users\saediga\Downloads\Input Mapping Sheet file_FI_Simple with Natural Lang 1.xlsx"
    
    excel_dict = create_nested_dict_from_excel(excel_file)
    # print("\nSource Dictionary:\n")
    # print(json.dumps(excel_dict, indent=4))

    # Process the dictionary
    output_dict = process_dictionary(excel_dict)
    print("\n///////////////////////////////////////////////////////////////Final Dictionary:\n")
    print(json.dumps(output_dict, indent=4))

if __name__ == "__main__":
    main()