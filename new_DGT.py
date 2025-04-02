import pandas as pd
import requests
import json
from cgitb import text
from tkinter.ttk import Separator
import uuid
import os
from PIL import Image
import subprocess
import shutil 
import re
import csv
import base64
import tempfile
import streamlit as st
from openai import AzureOpenAI
from io import StringIO, BytesIO
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

def loadJSONFile(file_path):
    docs=[]
    # Load JSON file
    with open(file_path) as file:
        data = json.load(file)

    # Iterate through 'pages'
    for page in data['pages']:
        parenturl = page['parenturl']
        pagetitle = page['pagetitle']
        indexeddate = page['indexeddate']
        snippets = page['snippets']
        metadata={"title":pagetitle}

        # Process snippets for each page
        for snippet in snippets:
            index = snippet['index']
            childurl = snippet['childurl']
            text = snippet['text']
            docs.append(Document(page_content=text, metadata=metadata))
    return docs 

def setup_embeddings(data):
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment="EmbadingModel",
                model="BayerBIGPT35Turbo",
                openai_api_key='836fea48dd7a402dbd8cf73f4dddb2e5',
                azure_endpoint="https://bayerbichatgpt4.openai.azure.com/",
                openai_api_type="azure",
            )
    
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    texts = text_splitter.split_documents(data)
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db2")
    vector_db.persist()
    return vector_db

def setup_qa_chain(prompt,llm, retriever):
    
    template = prompt+"""     
        {context}
        Question : {question}
        """
    # In the end of the code you have to write the following line of code: print(filename), filename should be replaced with the file name used for writing csv
    
    # Create a PromptTemplate object from the defined template
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Setup the RetrievalQA chain with the custom prompt template
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Ensure 'chain_type' is set to the appropriate value for your use case
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

def clean_code_block(code):
    # Remove leading and trailing whitespace (including extra lines)
            code = code.strip()
            # Remove the ```python at the start and ``` at the end
            if code.startswith("```python"):
                code = code[9:]  # Remove the first 9 characters ("```python\n")
            if code.endswith("```"):
                code = code[:-3]  # Remove the last 3 characters ("```")
            return code.strip()  # Strip again to clean up any remaining blank lines

import codecs

def convert_csv_to_ebcdic(input_csv, output_ebcdic):
    # Open the input CSV in text mode and read its contents
    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        csv_data = csv_file.read()
    
    # Write the data in EBCDIC encoding to the output file
    with codecs.open(output_ebcdic, 'w', encoding='cp500') as ebcdic_file:
        ebcdic_file.write(csv_data)          
   

def main():
    # Initialize session_state for messages if not already initialized
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    if "processing_done" not in st.session_state:
        st.session_state["processing_done"] = False  # Flag to track if processing is done

    st.title("Welcome to the Data Generation Tool")

    # Create a selection for the user to choose between Bulk load and Normal
    choice = st.radio("Choose an option", ("Bulk load", "Normal"))

    if choice == "Bulk load":
        # Run the existing bulk load code here
        # st.sidebar.title("Data generation")

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            if st.button("Process File"):
                with st.spinner("Reading..."):
                    df = pd.read_csv(uploaded_file)

                    for i in range(df.shape[0]):
                        path = df.loc[i, 'FilePath']
                        no_of_rec = df.loc[i, 'N0_Of_Records']
                        no_of_keys = df.loc[i, 'Number of Keys']
                        key_val = {}

                        for j in range(no_of_keys):
                            key_name = df.loc[i, f'Key{j+1} Name']
                            key_val[key_name] = int(df.loc[i, f"Key{j+1} ID"])

                        cus_prompt = df.loc[i, 'Detailed Prompt']
                        temp_df = pd.read_csv(path)

                        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as final_temp_file:
                            temp_df.to_csv(final_temp_file.name, index=False)
                            final_temp_file_path = final_temp_file.name

                        loader = CSVLoader(file_path=final_temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                        data = loader.load()
                        setup_embeddings(data)

                    st.session_state["processing_done"] = True  # Set the flag when processing is done
                    st.success("✅ Done", icon="✅")  # Corner message
                    st.session_state["file_processed"] = True  # Track that file is processed

        if st.session_state.get("file_processed", False):  
            # initialize embeddings object; for use with user query/input
            embed = AzureOpenAIEmbeddings(
                        azure_deployment="EmbadingModel",
                        model="BayerBIGPT35Turbo",
                        openai_api_key='836fea48dd7a402dbd8cf73f4dddb2e5',
                        azure_endpoint="https://bayerbichatgpt4.openai.azure.com/",
                        openai_api_type="azure",  
                        chunk_size=1024              
                    )
            vectorstore = Chroma(persist_directory="chroma_db2", embedding_function=embed)

            if st.button("Generate Data", key="generate_data_button"):
                df = pd.read_csv(uploaded_file)

                for i in range(df.shape[0]):
                    path = df.loc[i, 'FilePath']
                    no_of_rec = df.loc[i, 'N0_Of_Records']
                    no_of_keys = df.loc[i, 'Number of Keys']
                    key_val = {}
                    for j in range(no_of_keys):
                        key_name = df.loc[i, f'Key{j+1} Name']
                        key_val[key_name] = int(df.loc[i, f"Key{j+1} ID"])
                    cus_prompt = df.loc[i, 'Detailed Prompt']

                    dynamic_statement = f"Generate data for {no_of_rec} rows for all columns."
                    for key, value in key_val.items():
                        dynamic_statement += f" The values for the key {key} should start from {value} and increment by 1."
                    
                    prompt = f"{cus_prompt} {dynamic_statement} "
                    temp_df = pd.read_csv(path)

                    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as final_temp_file:
                        temp_df.to_csv(final_temp_file.name, index=False)
                        final_temp_file_path = final_temp_file.name  # This is the path you need

                    loader = CSVLoader(file_path=final_temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                    data = loader.load()
                    ext_data = []
                    for i in data:
                        ext_data.append((i).page_content)
                    com_data = " ".join(ext_data)

                    prompt += com_data

                    if prompt:
                        llm = AzureChatOpenAI(
                            temperature=0,
                            deployment_name="TDG-gpt-4o",
                            model_name="gpt-4o",
                            openai_api_key='5dTo0YIKUOzC2dhWENgEzfdnEG7qUD2ckhxDoxQfkXeIJCFbI1tQJQQJ99BAACYeBjFXJ3w3AAABACOGq1g9',
                            openai_api_version='2024-08-01-preview',
                            openai_api_base='https://testdatagenerator.openai.azure.com/',
                            openai_api_type='azure'
                        )
                        template = f""" You are an Intelligent faker code generator by using the sample data provided in the prompt. You will be provided the sample table you have to go through the structure of data and its columns and values.
                You have to generate a python code using Faker library for generating data similar data and save it to a csv file.
                For the description columns you have to generate similar meaningful descriptions mentioned in sample data.
                For the date it should be in similar range as it is in the context with date format as YYYY-MM-DD.
                Price or budget columns should be written with dollar symbol.
                Phone number should be realistic with country code (+1 for US) followed by 10 digit mobile number
                Email address should contains same customer first name and last name as mentioned in sample data.
                To avoid errors, faker code should take care of digits and characters while concatenating.
                \n Do not generate the data using random functions, please refer the sample data all columns data while generating faker code.
                \n Generated data should be like real life customer data.
                \n For the file name in the code always use "generated_data" the name.
                \n Do not use multi line comment in the code. Only output the python code nothing else.
                \n Always use datetime module to construct datetime object and then use in generating date values of columns with date. Follow the given syntax to generate date - fake.date_between(start_date='-30y', end_date=datetime(2033,1,12)).strftime('%Y-%m-%d')
                {prompt}"""
                        qa_chain = setup_qa_chain(template, llm, vectorstore.as_retriever())
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        result = qa_chain({"query": prompt})
                        msg = result['result']

                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        msg = clean_code_block(msg)
                        #st.code(msg)

                        subprocess.run(['python', '-c', msg], check=True)

                        path = path.split("\\")[-1]
                        data_df = pd.read_csv('generated_data.csv')
                        data_df.to_csv(f"generated_{path}")
                        input_csv = f'generated_{path}'

                        output_ebcdic = f'generated_ebc_{path}'
                        convert_csv_to_ebcdic(input_csv, output_ebcdic)
                        st.write(data_df)

    elif choice == "Normal":
        
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="normal_csv")
        
        if uploaded_file:
            if st.button("Process CSV"):
                with st.spinner("Processing..."):
                    df = pd.read_csv(uploaded_file)
                    temp_file_path = None
                    
                    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as temp_file:
                        df.to_csv(temp_file.name, index=False)
                        temp_file_path = temp_file.name
                    loader = CSVLoader(file_path=temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                    data = loader.load()
                    setup_embeddings(data)
                    st.session_state["file_processed"] = True
                    st.session_state["filename"] = uploaded_file.name
                    st.session_state["temp_file_path"] = temp_file_path
                    st.session_state["numerical_columns"] = df.select_dtypes(include=['number']).columns.tolist()
                    st.success(f"File {uploaded_file.name} processed successfully!")
        
        if st.session_state.get("file_processed", False):
            st.write(f"### Processed File: {st.session_state['filename']}")
            embed = AzureOpenAIEmbeddings(
                        azure_deployment="EmbadingModel",
                        model="BayerBIGPT35Turbo",
                        openai_api_key='836fea48dd7a402dbd8cf73f4dddb2e5',
                        azure_endpoint="https://bayerbichatgpt4.openai.azure.com/",
                        openai_api_type="azure",  
                        chunk_size=1024              
                    )
            vectorstore = Chroma(persist_directory="chroma_db2", embedding_function=embed)

            num_rows = st.number_input("Enter number of rows", min_value=1, step=1)
            num_keys = st.number_input("Enter number of keys", min_value=1, step=1)
            
            key_values = {}
            key_columns = st.session_state["numerical_columns"]
            
            st.write("### Define Keys")
            for i in range(int(num_keys)):
                col1, col2 = st.columns(2)
                with col1:
                    key_name = st.selectbox(f"Key {i+1} Name", key_columns, key=f"key_name_{i}")
                with col2:
                    key_start = st.number_input(f"Key {i+1} Starting Point", min_value=0, step=1, key=f"key_start_{i}")
                key_values[key_name] = key_start
            
            user_prompt = st.text_area("Enter your prompt",height = 200)
            
            if st.button("Process Data"):
                dynamic_statement = f"Generate data for {num_rows} rows for all columns."
                for key, value in key_values.items():
                    dynamic_statement += f" The values for the key {key} should start from {value} and increment by 1."
                
                final_prompt = f"{user_prompt} {dynamic_statement} "
                

                if final_prompt:
                        llm = AzureChatOpenAI(
                            temperature=0,
                            deployment_name="TDG-gpt-4o",
                            model_name="gpt-4o",
                            openai_api_key='5dTo0YIKUOzC2dhWENgEzfdnEG7qUD2ckhxDoxQfkXeIJCFbI1tQJQQJ99BAACYeBjFXJ3w3AAABACOGq1g9',
                            openai_api_version='2024-08-01-preview',
                            openai_api_base='https://testdatagenerator.openai.azure.com/',
                            openai_api_type='azure'
                        )
                        template = f""" You are an Intelligent faker code generator by using the sample data provided in the prompt. You will be provided the sample table you have to go through the structure of data and its columns and values.
                You have to generate a python code using Faker library for generating data similar data and save it to a csv file.
                For the description columns you have to generate similar meaningful descriptions mentioned in sample data.
                For the date it should be in similar range as it is in the context with date format as YYYY-MM-DD.
                Price or budget columns should be written with dollar symbol.
                Phone number should be realistic with country code (+1 for US) followed by 10 digit mobile number
                Email address should contains same customer first name and last name as mentioned in sample data.
                To avoid errors, faker code should take care of digits and characters while concatenating.
                \n Do not generate the data using random functions, please refer the sample data all columns data while generating faker code.
                \n Generated data should be like real life customer data.
                \n For the file name in the code always use "generated_data" the name.
                \n Do not use multi line comment in the code. Only output the python code nothing else.
                \n Always use datetime module to construct datetime object and then use in generating date values of columns with date. Follow the given syntax to generate date - fake.date_between(start_date='-30y', end_date=datetime(2033,1,12)).
                {final_prompt}"""
                        qa_chain = setup_qa_chain(template, llm, vectorstore.as_retriever())
                        st.session_state.messages.append({"role": "user", "content": final_prompt})
                        result = qa_chain({"query": final_prompt})
                        msg = result['result']

                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        msg = clean_code_block(msg)
                        st.code(msg)

                        subprocess.run(['python', '-c', msg], check=True)

                        #path = path.split("\\")[-1]
                        data_df = pd.read_csv('generated_data.csv')
                        data_df.to_csv(f"generated_{uploaded_file.name}")
                        input_csv = f'generated_{uploaded_file.name}'

                        output_ebcdic = f'generated_ebc_{uploaded_file.name}'
                        convert_csv_to_ebcdic(input_csv, output_ebcdic)
                        st.write(data_df)
                st.session_state["final_inputs"] = {
                    "num_rows": num_rows,
                    "num_keys": num_keys,
                    "key_values": key_values,
                    "user_prompt": user_prompt,
                    "dynamic_statement": dynamic_statement,
                    "final_prompt": final_prompt,
                }
                
                # st.write(f"**Generated Statement:** {final_prompt}")
                #st.success("Inputs saved successfully! Ready for further processing.")

                st.write("### What would you like to do next?")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Process Another File"):
                        # Reset session state to allow another file upload
                        st.session_state["file_processed"] = False
                        st.session_state["filename"] = None
                        st.session_state["temp_file_path"] = None
                        st.session_state["numerical_columns"] = None
                        st.session_state["final_inputs"] = None
                        st.experimental_rerun()

                with col2:
                    if st.button("Close Process"):
                        st.success("Process completed. You can now close the application.")

if __name__ == "__main__":
    main()
