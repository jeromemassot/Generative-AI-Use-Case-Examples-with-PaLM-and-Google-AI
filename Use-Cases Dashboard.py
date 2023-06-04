from google.oauth2 import service_account

import streamlit as st
import pandas as pd
import numpy as np

import os

#################################################################################
# GCP Credentials
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./petroglyphs-nlp.json"

#################################################################################
# Streamlit App

st.set_page_config(
    page_title="Generative AI Use-Cases with PaLM and Vertex AI"
)

# Title
st.title('Use-Cases with PaLM and Vertex AI')
st.subheader('Business value from Generative AI in Industry')

st.write("""
    This webapp gathers several Generative AI Use-Cases using PaLM and Vertex AI. 
    All the models used here are pre-trained models.
    """
)

st.info(""" No fine-tuning is required.
    Only prompts have been engineered to generate the results you will see.""")

st.write("""
    You can navigate through the examples using the sidebar on the left.
    There is a page per use-case.
    """
)

st.header('Use-Cases')

st.markdown("""
   - OCRed Images Text Curation
   - Data Curation
   - Knowledge Extraction
   - Anomaly Detection
   - Statistical Analysis
""")