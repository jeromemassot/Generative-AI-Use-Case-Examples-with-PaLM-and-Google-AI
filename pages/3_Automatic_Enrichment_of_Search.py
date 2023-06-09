from utils import (
    list_embedding_palm_models, list_generate_palm_models, 
    create_and_fill_glossary, init_sparse_encoder, search_index,
    answer
)
import streamlit as st
import pandas as pd
import pinecone

#################################################################################
# Automatic Semantic Search enrichment of existing glossary with PaLM and Pinecone

@st.cache_resource
def init_pinecone():
    """
    Initialize the pinecone connection
    :return: None
    """
    pinecone.init(api_key="963f18aa-f6dd-4fff-9175-9e24e8f2ab3c", environment='us-west1-gcp-free')


@st.cache_resource
def init_bm25_encoder(glossary, short_column):
    return init_sparse_encoder(glossary, short_column)


@st.cache_data
def load_glossary(uploaded_file):
    """
    Load the glossary from the csv file
    :return: the glossary as a pandas dataframe
    """
    glossary = pd.read_csv(uploaded_file, sep='|')
    glossary.columns = ['term',	'short', 'keywords', 'definition']
    return glossary


@st.cache_resource
def list_palm_models():
    """
    List the embedding models and generative models available in PaLM API
    :return: the list of the embedding models
    """
    return list_embedding_palm_models(), list_generate_palm_models()


# init the pinecone connection    
init_pinecone()

# list PaLM models
(embedding_models, embedding_dimensions), generate_models = list_palm_models()

# Title
st.title('Automatic Semantic enrichment of technical glossary with PaLM')
st.header('PaLM is used as a contexts reader to enrich technical glossary search')

st.write("""
    This use-case shows how to use quickly transform a technical glossary into a semantic search engine.
    """
)

st.subheader("Why this use-case?")

st.write("""
    The concept of "search engine" has evolved recently to become an "answer engine".
    An "answer engine" is able to understand the meaning of the question and provide an answer from a knowledge-base.
    The provided answer is generated by a Large Language Model (LLM) such as PaLM rather than extracted
    from retrieved documents.
    """
)

st.subheader("The pipeline")

st.write("""
    The pipeline is composed of 3 steps:
    - **Step 1**: upload the glossary as a csv file
    - **Step 2**: encode the glossary with PaLM embedding model
    - **Step 3**: index the glossary with Pinecone
    - **Step 4**: retrieve relevant contexts from the glossary with Pinecone depending on the user query
    - **Step 5**: generate an answer with PaLM generative model from these ranked contexts
    """
)

st.subheader("Glossary Upload")

st.sidebar.subheader("Glosssary Loading")
uploaded_file = st.sidebar.file_uploader("Choose an csv file", type=['csv'])

st.info("""
    You can upload your own technical glossary by using the uploader in the sidebar menu.
    The csv file should contains at least 2 columns to be used for ranking the contexts
    - **column 1**: the term defined in the glossary
    - **column 2**: the full definition which will be enconded as dense vectors
""")

glossary = None
if uploaded_file is not None:
    with st.expander("Show the glossary"):
        glossary = load_glossary(uploaded_file)
        st.dataframe(glossary, use_container_width=True, hide_index=True)

    term_column = st.sidebar.selectbox("Term column", options=glossary.columns)
    short_column = st.sidebar.selectbox("Short definition column", options=glossary.columns)
    definition_column = st.sidebar.selectbox("Full definition column", options=glossary.columns)
    keywords_column = st.sidebar.selectbox("Keywords column", options=glossary.columns)

st.subheader("Glossary Encoding")

st.write("""
    You can choose the model to encode the glossary. The model will be used to encode the full definitions.
    The short definition will be encoded as sparse vectors with a BM25 model available in the Pinecone library.
    """
)

st.info("Use the embedding model selector in the leff sidebar menu, and then click on the button to encode the glossary.")

st.sidebar.subheader("Glossary Sparse Encoding")
sparse_encode_glossary = st.sidebar.button("Init the sparse encoder from glossary")
if glossary is not None and len(short_column) > 0 and sparse_encode_glossary:
    bm25_encoder = init_bm25_encoder(glossary, short_column)
    st.session_state['bm25_encoder'] = bm25_encoder

st.sidebar.subheader("Glossary Dense Encoding")
embedding_model_name = st.sidebar.selectbox("Embedding model", options=embedding_models)

if embedding_model_name:
    embedding_dimension = embedding_dimensions[embedding_model_name]
    st.sidebar.write(f"Embedding dimension: {embedding_dimension}")

encode_glossary = st.sidebar.button("Encode the glossary")

if encode_glossary and glossary is not None:
    bm25_encoder = st.session_state['bm25_encoder']
    with st.spinner("Encoding the glossary..."):
        message_log = create_and_fill_glossary(
            glossary, 
            embedding_model_name, embedding_dimension, bm25_encoder,
            term_column, short_column, definition_column, keywords_column, 
            batch_size=64
        )
        if message_log == "The index already exists...":
            st.error(message_log)
        else:
            st.success("Glossary encoded")

st.subheader("Search Engine")

st.write("""Ask a question related to the technical terms that you have uploaded in the glossary.""")

st.info("Use the search bar below to ask a question.")

st.sidebar.subheader("Search Engine")

st.sidebar.write(f"Using index: {pinecone.list_indexes()[0]}")

with st.sidebar.form("Search Engine"):
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=3, step=1)
    st.caption("Number of contexts to retrieve from the glossary")

    alpha = st.number_input("Alpha", min_value=0, max_value=100, value=50, step=10)
    alpha = (100-alpha) / 100
    st.caption("Weight of the BM25 score in the final ranking (0-100)")

    deduplicate = st.checkbox("Deduplicate", value=True)
    st.caption("Deduplicate the contexts definition using similarity")

    reader_model = st.selectbox("Reader model", options=generate_models)
    st.caption("Model used to generate the answer")

    answer_button = st.form_submit_button("Answer")

question = st.text_input("Question", key="question")

if answer_button and question and len(question) > 0:
    ranked_contexts = search_index(
        question, st.session_state['bm25_encoder'], 
        embedding_model_name, top_k, alpha
    )

    matches = ranked_contexts["matches"]

    if deduplicate:
        unique_texts = []
        unique_matches = []
        for match in matches:
            if match["metadata"]["text"] not in unique_texts:
                unique_matches.append(match)
                unique_texts.append(match["metadata"]["text"])
        matches = list(unique_matches)

    if len(matches) > 0:
        context = ' '.join([match["metadata"]["text"] for match in matches])
        st.write("Found anwser:")
        st.write(
            answer(context, question, reader_model, nb_words=100)
        )

    with st.expander("Related terms and keywords"):
        st.multiselect(
            "Related terms", 
            options=[match["metadata"]["term"] for match in matches],
            default=sorted(match["metadata"]["term"] for match in matches)
        )
        st.multiselect(
            "Related keywords", 
            options=(k for match in matches for k in match["metadata"]["keywords"]),
            default=sorted(k for match in matches for k in match["metadata"]["keywords"])
        )
    
    with st.expander("Ranked contexts"):
        st.write(f"Top {top_k} ranked contexts:")
        for match in matches:
            st.write(match["id"])
            st.write(match["metadata"]["term"])
            st.write(match["metadata"]["keywords"])
            st.write(match["metadata"]["text"])
            st.write(match["score"])
