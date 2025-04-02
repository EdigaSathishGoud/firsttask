import warnings
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec
import os
from sentence_transformers import SentenceTransformer
api_key = 'c4f6971b-50d6-4c70-ac25-6b02638c31f1'
# configure client
pc = Pinecone(api_key=api_key)

index_name_sem = "semantic"

# connect to index
index = pc.Index(index_name_sem)

os.environ['PINECONE_API_KEY'] = "7b8a6cb3-f0ec-4cc3-bb40-88ad11ef6483"
index_name = "langchain-retrieval-augmentation-fast"


orders= pd.read_csv(r"C:\Users\saediga\OneDrive - Capgemini\Bayer\sumit\edited_ORDER_ITEMS.csv")
delivery=pd.read_csv(r"C:\Users\saediga\OneDrive - Capgemini\Bayer\sumit\edited_DELIVERY_ITEMS.csv")
billing=pd.read_csv(r"C:\Users\saediga\OneDrive - Capgemini\Bayer\sumit\edited_BILLING_ITEMS.csv")
col1=orders.columns 
col2=delivery.columns
col3=billing.columns 
#print(col1)
orders_schema= ", ".join(col1)
delivery_schema=", ".join(col2)
billing_schema=", ".join(col3)
firstn_orders=orders.head(5)
firstn_delivery=delivery.head(5)
firstn_billing=billing.head(5)
# print(orders_schema)
# print(delivery_schema)
# print(billing_schema)

def mappingToText(file):
    df = pd.read_csv(file, header=None)

    formatted_strings = []

    for index, row in df.iterrows():
        key, value = row[0], row[1]
        formatted_strings.append(f"{key} = {value}")

    result_string = ", ".join(formatted_strings)
    return result_string

order_mapping = pd.read_csv(r'C:\Users\saediga\OneDrive - Capgemini\Bayer\sumit\mapping_order1.csv')
delivery_mapping = pd.read_csv(r'C:\Users\saediga\OneDrive - Capgemini\Bayer\sumit\mapping_delivery1.csv')
billing_mapping = pd.read_csv(r'C:\Users\saediga\OneDrive - Capgemini\Bayer\sumit\mapping_billing1.csv')
# order_mapping = mappingToText('mapping_order.csv')
# delivery_mapping = mappingToText('mapping_delivery.csv')
# billing_mapping = mappingToText('mapping_billing.csv')
# print(order_mapping)
# print('\n\n')
# print(delivery_mapping)
# print('\n\n')
# print(billing_mapping)


text = f"""There are sql tables named orders with it's properties orders({orders_schema},
    Sales Order Document and Sales Order Item are primary keys),delivery with it's properties delivery({delivery_schema},
    Delivery Document and Delivery Item are primary keys , Sales Order Document and Sales Order Item are foreign keys referring Sales Order Document and Sales Order Item from orders),
    billing with it's properties billing({billing_schema},Billing Document and Billing Item are primary keys,Delivery Document and Delivery Item are foreign keys referring Delivery Document and Delivery Item from delivery) 
    You may have to do some transformations or conversions for data format by refering to the sample data for \n orders \n {firstn_orders} \n, \n delivery \n {firstn_delivery} \n and billing \n{firstn_billing}.
    \n Please use mapping carefully for converting the names from simplified to the original names. Mapping \n for orders {order_mapping} \n
    for delivery \n{delivery_mapping} \n for billing \n {billing_mapping}. \n     There are three tables which are orders, delivery and billing. You have to use the columns present in the schema for the associated tables for generating the sql query. Check carefully about the columns present for the table which is being referred to in the sql query and use accordingly.
    Perform joins properly by referring to the foreign and primary keys in the context mentioned above. Perform Joins only when required
    In the user query you can get column names which are simplified names of the columns in the schema. You have to refer to the mapping dataframes to obtain the original names of the columns which is there in the first column of the dataframe and next to that in the second column you will find the simplified names for the respective original names.
    Please try to use the simplified names as alias for the alias names.
                 """
# text = f"""There are sql tables named orders with it's properties orders({orders_schema},
#     Sales Order Document and Sales Order Item are primary keys),delivery with it's properties delivery({delivery_schema},
#     Delivery Document and Delivery Item are primary keys , Sales Order Document and Sales Order Item are foreign keys referring Sales Order Document and Sales Order Item from orders),
#     billing with it's properties billing({billing_schema},Billing Document and Billing Item are primary keys,Delivery Document and Delivery Item are foreign keys referring Delivery Document and Delivery Item from delivery) 
#     You may have to do some transformations or conversions for data format by refering to the sample data for \n orders \n {firstn_orders} \n, \n delivery \n {firstn_delivery} \n and billing \n{firstn_billing}.
#     \n Please use mapping carefully for converting the names from simplified to the original names. Mapping \n for orders {order_mapping} \n
#     for delivery \n{delivery_mapping} \n for billing \n {billing_mapping}. \n These mapping consists of original column names which are in schema and simplified names related to each original columns in the format 
#     "original names = simplified names" and you have to use original names in the sql query for the simplified names given in the prompt and you can use simplified names as alias in the generated sql query but in the sql query 
#     strictly use the names provided in the schemas for the column names.
#                  """
# print(text)


def getTextChunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks 



def getVectorStore(chunks):
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment="TextEmbeddingAda002",
                model="text-embedding-ada-002",
                openai_api_key='07b579590a6e48f2993bb34d2fd905f4',
                base_url="https://instgenaipoc.openai.azure.com/",
                openai_api_type="azure",
            )  
    vector_store = FAISS.from_texts(chunks, embedding=embeddings) 
    # vector_store.save_local('vec_store')
    return vector_store

def createPineVector():
    docs = getTextChunks(text)
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment="TextEmbeddingAda002",
                model="text-embedding-ada-002",
                openai_api_key='07b579590a6e48f2993bb34d2fd905f4',
                base_url="https://instgenaipoc.openai.azure.com/",
                openai_api_type="azure",
            )
    PineconeVectorStore.from_texts(docs, embeddings, index_name=index_name)

# createPineVector()

def getPineVector():
    embeddings = AzureOpenAIEmbeddings(
                azure_deployment="TextEmbeddingAda002",
                model="text-embedding-ada-002",
                openai_api_key='07b579590a6e48f2993bb34d2fd905f4',
                base_url="https://instgenaipoc.openai.azure.com/",
                openai_api_type="azure",
            )
    return PineconeVectorStore(index_name=index_name,embedding=embeddings)



# def get_conversational_chain(vector_store):
#     llm = AzureChatOpenAI(deployment_name="chatgpt45turbo",
#                       model_name="gpt-35-turbo",
#                       openai_api_base="https://instgenaipoc.openai.azure.com/",
#                       openai_api_version="2023-05-15",
#                       openai_api_key="07b579590a6e48f2993bb34d2fd905f4",
#                       openai_api_type="azure")  
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     template = """Generate SQL query for the given tables in the {context}.
#     In the context you will find the mapping dataframes for each table.     There are three tables which are orders, delivery and billing. You have to use the columns present in the schema for the associated tables for generating the sql query. Check carefully about the columns present for the table which is being referred to in the sql query and use accordingly.
#     Perform joins properly by referring to the foreign and primary keys in the context mentioned above.
#     In the user query you can get column names which are simplified names of the columns in the schema. You have to refer to the mapping dataframes to obtain the original names of the columns which is there in the first column of the dataframe and next to that in the second column you will find the simplified names for the respective original names.
#     Please try to use the simplified names as alias for the alias names.
#         To denote the column name in query use ` at the starting and ending of column name also the sql code should not have ``` in beginning or at the end of sql output and there shouldn't be any explanation for the generated query.
#         {context}
#         {question}"""

#     PROMPT = PromptTemplate(
#         input_variables=["context", "question"], 
#         template=template
#     )

#     conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,combine_docs_chain_kwargs={'prompt': PROMPT})
#     return conversation_chain



def get_conversational_chain(vector_store):
    llm = AzureChatOpenAI(deployment_name="chatgpt45turbo",
                      model_name="gpt-35-turbo",
                      openai_api_base="https://instgenaipoc.openai.azure.com/",
                      openai_api_version="2023-05-15",
                      openai_api_key="07b579590a6e48f2993bb34d2fd905f4",
                      openai_api_type="azure")  
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    general_system_template = r""" 
    Generate SQL query for the question.
    In the mapping you will find the original names = simplified names type of data which is the actual names of the columns and the simplified names next to that. Simplified names can be used in the
    question but during the sql query generation you have to use original names for that respective simplified names. 
    One important rule is you have to strictly use the columns which are there for the table which means you can't use columns of one table and the table name for other table. Don't do these mistakes.
    To denote the column name in query use ` at the starting and ending of column name also the sql code should not have ``` in beginning or at the end of sql output and there shouldn't be any explanation for the generated query.
    see this example for avoiding the mistake - user question - what is the bill type for each source system ? 
    answer - SELECT DISTINCT SOURSYSTEM, BILL_TYPE FROM billing
    the above answer is wrong because there is not bill_type column in billing table and the columns which are used in the sql is in delivery table so use the column and table name wisely.
        {context}
        {question} 
        """
    # general_system_template = f""" 
    # Generate SQL query for the given tables in the {context}.
    # In the context you will find the mapping dataframes for each table.     There are three tables which are orders, delivery and billing. You have to use the columns present in the schema for the associated tables for generating the sql query. Check carefully about the columns present for the table which is being referred to in the sql query and use accordingly.
    # Perform joins properly by referring to the foreign and primary keys in the context mentioned above.
    # In the user query you can get column names which are simplified names of the columns in the schema. You have to refer to the mapping dataframes to obtain the original names of the columns which is there in the first column of the dataframe and next to that in the second column you will find the simplified names for the respective original names.
    # Please try to use the simplified names as alias for the alias names.
    #     To denote the column name in query use ` at the starting and ending of column name also the sql code should not have ``` in beginning or at the end of sql output and there shouldn't be any explanation for the generated query.
    #     {context}
    #     """
    general_user_template = "```{question}```"
    messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,combine_docs_chain_kwargs={'prompt': qa_prompt})
    return conversation_chain


# vec_store = getPineVector()
# query = "what is sales order document"
# docs1 = vec_store.similarity_search(query)
# print(docs1[0].page_content)



def initialize_conversation():
    # chunks = getTextChunks(text)
    # vec_store = getVectorStore(chunks)
    vec_store = getPineVector()
    st.session_state.conversation = get_conversational_chain(vec_store)

def main():
    st.set_page_config(
        page_title="chatbot",
        page_icon="🍄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    faqs = [
        "Sum of delivery items for each customer group",
        "what is the bill type for each source system",
        "Sum of delivery number for delivery group"
    ]

    # Initialize session state variables if not present
    if "conversation" not in st.session_state:
        initialize_conversation()

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    user_question = st.text_input("Chatbot")  # Input field for user's question
    response={}
    #load the model
    model1 = SentenceTransformer('all-mpnet-base-v2')
    if user_question:
        score=0
        xq=model1.encode([user_question]).tolist()
        results = index.query(vector=xq,top_k=1,include_metadata=True)

        for result in results['matches']:
            score,ans=round(result['score'], 2),list(result['metadata'].values())[0]
    
        
        if score>0.8:
            st.write(f"Found in History with score {score}")
            response['content'] = ans
            st.session_state.chatHistory.append({"content": user_question, "is_user": True})
            st.session_state.chatHistory.append({"content": ans, "is_user": False})
            print("matched response", response)
            st.markdown(response) # Store the conversation history

        else:
            response = st.session_state.conversation({'question': user_question})  # Start a conversation with a user's question
            print("********",response)
            index.upsert(vectors=[{'id':user_question, 'values': model1.encode(user_question).tolist(),'metadata': {'Question': user_question, 'Answer': response['answer']}}])
            # st.session_state.chatHistory = response['chat_history']  # Store the conversation history
            st.session_state.chatHistory.extend(response['chat_history'])

                        # Normalize the LLM response format
            normalized_chat_history = []
            for message in response['chat_history']:
                if isinstance(message, HumanMessage):
                    normalized_chat_history.append({"content": message.content, "is_user": True})
                elif isinstance(message, AIMessage):
                    normalized_chat_history.append({"content": message.content, "is_user": False})

            st.session_state.chatHistory.extend(normalized_chat_history)

    # Display the chat history
    if st.session_state.chatHistory:
        print("chat history=============",st.session_state.chatHistory)
        for i, message in enumerate(st.session_state.chatHistory):
            
            if i % 2 == 0:
                # User's message (right-aligned)
                st.markdown(f'<div style="color: black; background-color: #d6f1f1; padding: 5px; border-radius: 5px; margin: 0 30% 10px 0;">Question: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                # Bot's message (left-aligned)
                st.markdown(f'<div style="color: black; background-color: #f5f5f5; padding: 5px; border-radius: 5px; margin: 0 0 10px 30%;">Response: {message["content"]}</div>', unsafe_allow_html=True)
def main1():
    st.set_page_config(
        page_title="chatbot",
        page_icon="🍄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    check = index.list(prefix='', namespace="")
    # print(check)
    lst = list(check)
    # print(lst)
    id = lst[0]
    # print(id)
    diction={}
    faq = []
    for i in id:
        # print(i)
        vectors=index.query(id=i,top_k=1,include_metadata=True)
        for vector in vectors['matches']:
            id,ans,counter=vector['id'],list(vector['metadata'].values())[0],list(vector['metadata'].values())[2]
            diction[id]=counter

    # print(diction)
    #Make the dictionary descending order of the counter
    sorted_dict = sorted(diction.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_dict)
    new_dict = {key: value for key, value in sorted_dict}
    # Printing the sorted dictionary
    for id, count in new_dict.items():
        # print(id, ' is asked ', count,' times')
        faq.append(id)

    
    # print(faq)

    # faqs = [
    #     "Sum of delivery items for each customer group",
    #     "what is the bill type for each source system",
    #     "Sum of delivery number for delivery group"
    # ]

    # Initialize session state variables if not present
    if "conversation" not in st.session_state:
        initialize_conversation()

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    user_question = st.text_input("Chatbot")  # Input field for user's question
    response = {}
    # Load the model
    model1 = SentenceTransformer('all-mpnet-base-v2')
    
    if user_question:
        score = 0
        xq = model1.encode([user_question]).tolist()
        results = index.query(vector=xq, top_k=1, include_metadata=True)

        for result in results['matches']:
            # score, ans = round(result['score'], 2), list(result['metadata'].values())[0]
            id,score,ans,counter=result['id'],round(result['score'], 2),list(result['metadata'].values())[0],list(result['metadata'].values())[2]

        if score > 0.8:
            
            st.write(f"Found in History with score {score}")
            response['content'] = ans
            index.update(id=id, set_metadata={"count": counter+1})
            st.session_state.chatHistory.append({'role': 'user', 'content': user_question})
            st.session_state.chatHistory.append({'role': 'bot', 'content': response['content']})
            st.markdown(f'<div style="color: black; background-color: #f5f5f5; padding: 5px; border-radius: 5px; margin: 0 0 10px 30%;">Response: {response["content"]}</div>', unsafe_allow_html=True)
        else:
            response = st.session_state.conversation({'question': user_question})  # Start a conversation with a user's question
            print("********", response)
            # index.upsert(vectors=[{'id': user_question, 'values': model1.encode(user_question).tolist(), 'metadata': {'Question': user_question, 'Answer': response['answer']}}])
            index.upsert(vectors=[{'id':user_question, 'values': model1.encode(user_question).tolist(),'metadata': {'Question': user_question, 'Answer': response,'count': 1}}])
            st.session_state.chatHistory.append({'role': 'user', 'content': user_question})
            st.session_state.chatHistory.append({'role': 'bot', 'content': response['answer']})

    # Display the chat history
    if st.session_state.chatHistory:
        # print(st.session_state.chatHistory)
        for message in st.session_state.chatHistory:
            # print('--------------------\n\n' , st.session_state.chatHistory)
            if message['role'] == 'user':
                st.markdown(f'<div style="color: black; background-color: #d6f1f1; padding: 5px; border-radius: 5px; margin: 0 30% 10px 0;">Question: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: black; background-color: #f5f5f5; padding: 5px; border-radius: 5px; margin: 0 0 10px 30%;">Response: {message["content"]}</div>', unsafe_allow_html=True)






def main2():
    st.set_page_config(
        page_title="chatbot",
        page_icon="🍄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Fetching the list of vectors
    check = index.list(prefix='', namespace="")
    lst = list(check)
    id_list = lst[0]
    diction = {}
    faq = []

    for i in id_list:
        vectors = index.query(id=i, top_k=1, include_metadata=True)
        for vector in vectors['matches']:
            id, ans, counter = vector['id'], list(vector['metadata'].values())[0], list(vector['metadata'].values())[2]
            diction[id] = counter

    # Sort the dictionary by counter in descending order
    sorted_dict = sorted(diction.items(), key=lambda item: item[1], reverse=True)
    new_dict = {key: value for key, value in sorted_dict}

    for id, count in new_dict.items():
        faq.append(id)

    # [('Tell me about AI', 2.0), ('What is YOLO in AI', 1.0), ('Describe YOLO', 0.0), ('Who is Virat kohili', 0.0)]

    # Initialize session state variables if not present
    if "conversation" not in st.session_state:
        initialize_conversation()

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    # Sidebar button and dropdown
    if st.sidebar.button("Show FAQs"):
        st.session_state.show_faqs = True

    if "show_faqs" in st.session_state and st.session_state.show_faqs:
        selected_question = st.sidebar.selectbox("Select a FAQ", faq)
        if st.sidebar.button("Ask FAQ"):
            user_question = selected_question
        else:
            user_question = st.text_input("Chatbot")  # Input field for user's question
    else:
        user_question = st.text_input("Chatbot")  # Input field for user's question

    response = {}
    # Load the model
    model1 = SentenceTransformer('all-mpnet-base-v2')

    if user_question:
        score = 0
        xq = model1.encode([user_question]).tolist()
        results = index.query(vector=xq, top_k=1, include_metadata=True)

        for result in results['matches']:
            id, score, ans, counter = result['id'], round(result['score'], 2), list(result['metadata'].values())[0], list(result['metadata'].values())[2]

        if score > 0.8:
            st.write(f"Found in History with score {score}")
            response['content'] = ans
            index.update(id=id, set_metadata={"count": counter + 1})
            st.session_state.chatHistory.append({'role': 'user', 'content': user_question})
            st.session_state.chatHistory.append({'role': 'bot', 'content': response['content']})
            # st.markdown(f'<div style="color: black; background-color: #f5f5f5; padding: 5px; border-radius: 5px; margin: 0 0 10px 30%;">Response: {response["content"]}</div>', unsafe_allow_html=True)
        else:
            response = st.session_state.conversation({'question': user_question})  # Start a conversation with a user's question
            print("********", response)
            index.upsert(vectors=[{'id': user_question, 'values': model1.encode(user_question).tolist(), 'metadata': {'Question': user_question, 'Answer': response, 'count': 1}}])
            st.session_state.chatHistory.append({'role': 'user', 'content': user_question})
            st.session_state.chatHistory.append({'role': 'bot', 'content': response['answer']})

    # Display the chat history
    if st.session_state.chatHistory:
        for message in st.session_state.chatHistory:
            if message['role'] == 'user':
                st.markdown(f'<div style="color: black; background-color: #d6f1f1; padding: 5px; border-radius: 5px; margin: 0 30% 10px 0;">Question: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: black; background-color: #f5f5f5; padding: 5px; border-radius: 5px; margin: 0 0 10px 30%;">Response: {message["content"]}</div>', unsafe_allow_html=True)





def main3():
    st.set_page_config(
        page_title="chatbot",
        page_icon="🍄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    #user_question = st.session_state.user_question or st.text_input("Chatbot", key="user_input")
    username = st.sidebar.text_input('enter you name')

    # Initialize Pinecone
    # pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
    # index = pinecone.Index(os.getenv("PINECONE_INDEX"))

    # Fetching the list of vectors
    check = index.list(prefix='', namespace=username)
    lst = list(check)
    id_list = lst[0]
    diction = {}
    faq = []

    for i in id_list:
        vectors = index.query(id=i, top_k=1, include_metadata=True, namespace=username)
        for vector in vectors['matches']:
            id, ans, counter = vector['id'], list(vector['metadata'].values())[0], list(vector['metadata'].values())[2]
            diction[id] = counter

    # Sort the dictionary by counter in descending order
    sorted_dict = sorted(diction.items(), key=lambda item: item[1], reverse=True)
    new_dict = {key: value for key, value in sorted_dict}

    for id, count in new_dict.items():
        faq.append(id)

    # Initialize session state variables if not present
    if "conversation" not in st.session_state:
        initialize_conversation()

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    # Sidebar button and dropdown
    if st.sidebar.button("Show FAQs"):
        st.session_state.show_faqs = True

    if "show_faqs" in st.session_state and st.session_state.show_faqs:
        st.session_state.selected_question = st.sidebar.selectbox("Select a FAQ", faq, key="faq_selectbox", index=None, placeholder="Choose an question")
        if st.sidebar.button("Ask FAQ"):
            st.session_state.user_question = st.session_state.selected_question

    user_question = st.session_state.user_question or st.text_input("Chatbot", key="user_input")
    # user_question = st.text_input("Chatbot", value=st.session_state.get("selected_question", ""))  # Input field for user's question

    response = {}
    # Load the model
    model1 = SentenceTransformer('all-mpnet-base-v2')

    if user_question:
        score = 0
        xq = model1.encode([user_question]).tolist()
        results = index.query(vector=xq, top_k=1, include_metadata=True, namespace=username)

        for result in results['matches']:
            id, score, ans, counter = result['id'], round(result['score'], 2), list(result['metadata'].values())[0], list(result['metadata'].values())[2]

        if score > 0.8:
            st.write(f"Found in History with score {score}")
            response['content'] = ans
            index.update(id=id, set_metadata={"count": counter + 1}, namespace=username)
            st.session_state.chatHistory.append({'role': 'user', 'content': user_question})
            st.session_state.chatHistory.append({'role': 'bot', 'content': response['content']})
        else:
            response = st.session_state.conversation({'question': user_question})  # Start a conversation with a user's question
            print("********", response)
            index.upsert(vectors=[{'id': user_question, 'values': model1.encode(user_question).tolist(), 'metadata': {'Question': user_question, 'Answer': response, 'count': 1}}], namespace=username )
            st.session_state.chatHistory.append({'role': 'user', 'content': user_question})
            st.session_state.chatHistory.append({'role': 'bot', 'content': response['answer']})
        
        st.session_state.user_question = ""  # Reset user_question after it's processed

    # Display the chat history
    if st.session_state.chatHistory:
        user_question = st.session_state.user_question or st.text_input("Chatbot", key="user_input")
        for message in st.session_state.chatHistory:
            if message['role'] == 'user':
                st.markdown(f'<div style="color: black; background-color: #d6f1f1; padding: 5px; border-radius: 5px; margin: 0 30% 10px 0;">Question: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: black; background-color: #f5f5f5; padding: 5px; border-radius: 5px; margin: 0 0 10px 30%;">Response: {message["content"]}</div>', unsafe_allow_html=True)




if __name__ == "__main__":
    main3()
    # createPineVector()




# 20 columns for each mapping
# question from the list

# do testing after adding columns in the prompts
# integrate semantic search
# store user id and store questions related to it
# store user id match them and give top 10 question

# add count +1 to question which is getting match





