from utils import detect_text, instruct_agent
import streamlit as st


#################################################################################
# Knowledge Extraction from challenging OCRed Texts with PaLM

# Title
st.title('Knowledge Extraction from Challenging OCRed Texts with PaLM')
st.header('PaLM replacing Regex and heuristics in OCR post-processing')

st.write("""
    This use-case shows how to use PaLM and Vertex AI to extract knowledge from OCRed texts.
    """
)

st.subheader("Why this use-case?")

st.write("""
    It could happen that the text extracted from a document is not perfect due to imperfect traditional post-processing
    techniques. The extraction of clean knowledge from this text is then challenging.
    - Vision AI pre-trained model is able to extract the text from the document with a very high accuracy.
    - PaLM goes one step further by curating the text and reconstructing Knowledge from it.
    """
)

st.subheader("The pipeline")

st.write("""
    The pipeline is composed of 3 steps:
    - **Step 1**: OCR of texts contained in the document pages with pre-trained Vision AI
    - **Step 2**: ingestion of the json file into PaLM
    - **Step 3**: curation and extraction of know ledge from the text with PaLM
    """
)

st.subheader("Image upload")

st.sidebar.subheader("Document page Loading")
uploaded_file = st.sidebar.file_uploader("Choose an page file", type=['png', 'jpg', 'jpeg'])

st.info("You can upload your own page of document by using the uploader in the sidebar menu.")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Page.')

st.subheader("Extracting Texts from the Page with GCP Vision AI")
st.info("You can detect the texts from the image by clicking the button in the sidebar menu.")

st.sidebar.subheader("Texts Extraction (Vision AI)")
detect_text_button = st.sidebar.button("Detect Text")

if detect_text_button:

    context = ""
    if uploaded_file is not None:
            content = uploaded_file.read()
            blocks = detect_text(content)
            for block in blocks:
                context += block + ' '
            st.session_state['context'] = context
            st.session_state['blocks'] = blocks

blocks = st.session_state.get('blocks', None)
if blocks and len(blocks)>0:
    with st.expander("Blocks Extracted from the Page"):
        for i, block in enumerate(blocks):
            st.write(f'block #{i+1}')
            st.caption(block)
else:
    st.warning("Please upload an page first.")

st.subheader("Extract Knowledge from the Texts with PaLM")
st.info("Enter the instructions that the agent will follow to get more value from the extracted text.")

instructions = st.text_area("Instructions", "Clean the text and Extract relevant information from it. In particular, reconstruct knowledge from table content.")
extract_button = st.button("Knowledge Extraction with PaLM")

context = st.session_state.get('context', None)

if context and len(context)>0 and extract_button:

    enriched_text = instruct_agent(context, instructions, nb_words=500)
    if enriched_text and len(enriched_text)>0:
        if instructions.lower().find("json")>=0:
            st.json(enriched_text)
        st.caption(enriched_text)
    else:
        st.caption("Nothing has been found Sorry")