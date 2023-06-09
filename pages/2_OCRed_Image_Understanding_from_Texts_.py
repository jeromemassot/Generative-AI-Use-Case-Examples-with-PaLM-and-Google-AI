from utils import detect_text, instruct_agent
import streamlit as st


#################################################################################
# OCRed Image Texts Curation with PaLM

# Title
st.title('Generating Insights from Texts extracted from Image')
st.header('PaLM replacing trained image captioning models')

st.write("""
    This use-case shows how to use PaLM and Vertex AI to create value from texts extracted from an image.
    """
)

st.subheader("Why this use-case?")

st.write("""
    Very often, the text extracted from images is not perfect due to imperfect traditional post-processing
    techniques. 
    - Vision AI pre-trained model is able to extract the text from the image with a very high accuracy.
    - PaLM goes one step further by curating the text and generating insights from it.
    """
)

st.subheader("The pipeline")

st.write("""
    The pipeline is composed of 3 steps:
    - **Step 1**: OCR of the images text with pre-trained Vision AI
    - **Step 2**: ingestion of the json file into PaLM
    - **Step 3**: curation and analysis of the text with PaLM
    """
)

st.subheader("Image upload")

st.sidebar.subheader("Image Loading")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

st.info("You can upload your own image by using the uploader in the sidebar menu.")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.')

st.subheader("Extracting Texts from the Image with GCP Vision AI")
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
    with st.expander("Texts Extracted from the Image"):
        st.caption(blocks)
else:
    st.warning("Please upload an image first.")

st.subheader("Analysis and Enrichment with PaLM")
st.info("Enter the instructions that the agent will follow to get more value from the extracted text.")

instructions = st.text_area("Instructions", "What is this figure about? Use a maximum of 100 words. In particular, describe the elements present in the figure if possible.")
enrich_text_button = st.button("Describe Figure with PaLM")

context = st.session_state.get('context', None)
if context and len(context)>0 and enrich_text_button:
    enriched_text = instruct_agent(context, instructions, nb_words=500)
    if enriched_text and len(enriched_text)>0:
        if instructions.lower().find("json")>=0:
            st.json(enriched_text)
        st.caption(enriched_text)
    else:
        st.caption("Nothing has been found Sorry")