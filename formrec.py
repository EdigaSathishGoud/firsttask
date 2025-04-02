import json
import pandas as pd

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient


endpoint = "https://sathishformrec.cognitiveservices.azure.com/"
key = "723085f8f7524b50b46f986153878012"

with open(r"C:\Users\saediga\Downloads\hp_support_pdf.pdf", "rb") as fd:
    document = fd.read()

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

poller = document_analysis_client.begin_analyze_document(
        "prebuilt-read", document)
result = poller.result()