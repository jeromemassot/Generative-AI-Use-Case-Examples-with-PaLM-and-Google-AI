from utils import get_entities
import streamlit as st
import pandas as pd

#################################################################################
# Automatic Entities Recognition and Consolidation with PaLM

# Title
st.title('Identifying and Consolidating Entities with PaLM')
st.header('PaLM replacing trained models for NER and entity resolution')

st.write("""
    This use-case shows how to use PaLM to extract entities from a text and consolidate them.
    """
)

st.subheader("Why this use-case?")

st.write("""
    Very often, it is challenging to extract entities from a text without an pricey trained model. And even with a
    trained model, the entities are not consolidated, causing duplicates and inconsistencies in the knowledge-base.
    - PaLM is able to extract entities from a text, without the need of a trained model.
    - PaLM can use a given knowledge-base to consolidate the entities.
    """
)

st.subheader("The pipeline")

st.write("""
    The pipeline is composed of 2 steps:
    - **Step 1**: controlled vocabulary is entered by the user
    - **Step 2**: named entities are recognized and consolidated with PaLM
    """
)

st.subheader("Controlled Vocabulary upload")

st.sidebar.subheader("Vocabulary Loading")
uploaded_file = st.sidebar.file_uploader("Choose an csv file", type=['csv'])

st.info("""
    You can upload your own controlled vocabulary by using the uploader in the sidebar menu.
    The csv file should have 2 columns:
    - **column 1**: the entities (words or ngams)
    - **column 2**: the entity type
""")

if uploaded_file is not None:
    vocabulary = pd.read_csv(uploaded_file)
    categories = vocabulary['category'].unique()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(vocabulary[:9], use_container_width=True, hide_index=True)
    with col2:
        st.dataframe(vocabulary[9:], use_container_width=True, hide_index=True)

st.subheader("Extracting Entities from the text with PaLM")
st.info("""
    You can type any text in the text area below and click the button in the sidebar menu to extract entities.
    It it corresponds to a recognized entity, the original text is mapped to the reference entity text.
    This ensures that the entities are consolidated (entity resolution).
""")

original_text = st.text_area("Enter your text here", height=150)

st.sidebar.subheader("Named Entities Extraction")
extract_entities_button = st.sidebar.button("Extract Entities")

if extract_entities_button and len(vocabulary) > 0 and len(original_text) > 0:
    extracted_entities = get_entities(original_text, vocabulary.to_csv(index=False))
    try:
        extracted_entities = {k: v for k, v in extracted_entities.items() if v['category'] in categories}
        st.write("PaLM has extracted and consolidated the following entities:")
        st.json(extracted_entities)
    except TypeError:
        st.error("Sorry something went wrong in the formating of the response...")
        st.write(extracted_entities)